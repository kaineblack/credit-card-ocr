import os
import cv2
import imutils
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
from imutils import contours
import numpy as np


class CreditCardOCR:
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    sq_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    '''
    This class is used to parse the 16 digit card number
    from an image showing the front of a credit card. 
    '''
    def __init__(self, path_to_image, path_to_reference_image):
        assert os.path.exists(path_to_image), "Could not find the image in the path provided."
        assert os.path.exists(path_to_reference_image), "Could not find the reference image in the path provided."
        self.image = cv2.imread(path_to_image)
        self.image = imutils.resize(self.image, width=300)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        self.reference_image = cv2.imread(path_to_reference_image)
        if self.image is None:
            raise TypeError("Image could not be loaded in. Please double check the path and image file.")
        if self.reference_image is None:
            raise TypeError("Reference image could not be loaded in. Please double check the path and image file.")
    

    def prepare_reference_digits(self):
        # find contours in the OCR-A image (i.e,. the outlines of the digits)
        # sort them from left to right, and initialize a dictionary to map
        # digit name to the ROI
        ref = self.reference_image
        ref = imutils.resize(ref, 300)
        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY)[1]
        refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        refCnts = imutils.grab_contours(refCnts)
        refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]
        digits = {}
        # loop over the OCR-A reference contours
        for (i, c) in enumerate(refCnts):
            # compute the bounding box for the digit, extract it, and resize
            # it to a fixed size
            (x, y, w, h) = cv2.boundingRect(c)
            roi = ref[y:y + h, x:x + w]
            roi = cv2.resize(roi, (57, 88))
            # update the digits dictionary, mapping the digit name to the ROI
            digits[i] = roi

        return digits


    def show_original_image(self):
        original_image = self.image
        plt.axis('off')
        plt.imshow(original_image, vmin=0, vmax=255)
        plt.show()


    def convert_image_to_grayscale(self, image):
        grayscale_image = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2GRAY)

        return grayscale_image


    def show_grayscale_image(self):
        grayscale_image = self.convert_image_to_grayscale(self.image)
        # show the image in grayscale
        plt.axis('off')
        plt.imshow(grayscale_image, cmap='gray', vmin=0, vmax=255)
        plt.show()


    def apply_tophat_morphological_transform(self, image):
        # the tophat morphological operation reveals light regions on a dark background
        tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, self.rect_kernel)

        return tophat


    def show_tophat_image(self):
        tophat_image = self.apply_tophat_morphological_transform(
            self.convert_image_to_grayscale(self.image)
        )
        # show the image in grayscale
        plt.axis('off')
        plt.imshow(tophat_image, cmap='gray', vmin=0, vmax=255)
        plt.show()
    

    def apply_sobel_gradient(self, image):
        gradX = cv2.Sobel(
            self.apply_tophat_morphological_transform(
                self.convert_image_to_grayscale(self.image)
            ), 
            ddepth=cv2.CV_32F, 
            dx=1, 
            dy=0, 
            ksize=-1
        )
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
        gradX = gradX.astype("uint8")

        return gradX


    def show_sobel_image(self):
        sobel_image = self.apply_sobel_gradient(self.image.copy())
        # show the image in grayscale
        plt.axis('off')
        plt.imshow(sobel_image, cmap='gray', vmin=0, vmax=255)
        plt.show()


    def close_sobel_gradient(self, image):
        gradX = cv2.morphologyEx(image.copy(), cv2.MORPH_CLOSE, self.rect_kernel)
        thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # apply a second closing operation to the binary image, again
        # to help close gaps between credit card number regions
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.sq_kernel)

        return thresh


    def show_closed_sobel_image(self):
        closed_sobel = self.close_sobel_gradient(
            self.apply_sobel_gradient(
                self.convert_image_to_grayscale(self.image.copy())
        ))
        # show the image in grayscale
        plt.axis('off')
        plt.imshow(closed_sobel, cmap='gray', vmin=0, vmax=255)
        plt.show()


    def create_contours(self, image):
        cnts = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        return cnts

    
    def create_bounding_boxes(self, contours):
        boxes = []
        for c in contours:
            # translate the contours into simpler bounding box
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)

            # we can select the desired bounding boxes for the digits by
            # looking at a combination of the aspect ratio & height & width
            if (ar > 2.5 and ar < 4.0) and ((w > 40 and w < 55) and (h > 10 and h < 20)):
                # also adding a bit of padding/extra room to the bounding rectangle
                boxes.append((x, y, w, h))
        
        return boxes


    def match_digits_in_boxes(self, grayscale_image, bounding_boxes, digits):
        # sort the bounding boxes form left to right
        boxes = sorted(bounding_boxes, key=lambda x:x[0])

        # initialize output list
        output = []
        final_image = self.image.copy()

        # loop over the 4 groupings of 4 digits (hopefully)
        for gX, gY, gW, gH in boxes:
            # initialize the list of group digits
            groupOutput = []
            # extract the group ROI of 4 digits from the grayscale image,
            # then apply thresholding to segment the digits from the
            # background of the credit card
            group = grayscale_image[gY - 4:gY + gH + 4, gX - 4:gX + gW + 4]
            group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            # detect the contours of each individual digit in the group,
            # then sort the digit contours from left to right
            digitCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            digitCnts = imutils.grab_contours(digitCnts)
            digitCnts = contours.sort_contours(digitCnts,method="left-to-right")[0]
            # loop over the digit contours
            for c in digitCnts:
                # compute the bounding box of the individual digit, extract
                # the digit, and resize it to have the same fixed size as
                # the reference OCR-A images
                (x, y, w, h) = cv2.boundingRect(c)
                roi = group[y:y + h, x:x + w]
                roi = cv2.resize(roi, (57, 88))
                # initialize a list of template matching scores	
                scores = []
                # loop over the reference digit name and digit ROI
                for (digit, digitROI) in digits.items():
                    # apply correlation-based template matching, take the
                    # score, and update the scores list
                    result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
                    _, score, _, _ = cv2.minMaxLoc(result)
                    scores.append(score)
            
                # the classification for the digit ROI will be the reference
                # digit name with the *largest* template matching score
                groupOutput.append(str(np.argmax(scores)))

            # draw the digit classifications around the group
            cv2.rectangle(final_image, (gX - 4, gY - 4), (gX + gW + 4, gY + gH + 4), (255, 0, 0), 2)
            cv2.putText(final_image, "".join(groupOutput), (gX, gY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 0, 0), 2)
            # update the output digits list
            output.append(groupOutput)

        return (final_image, output)

        
    def process_credit_card(self):
        # digits for comparison
        digits = self.prepare_reference_digits()

        # image processing steps
        grayscale_image = self.convert_image_to_grayscale(self.image.copy())
        tophat_image = self.apply_tophat_morphological_transform(grayscale_image)
        sobel_image = self.apply_sobel_gradient(tophat_image)
        closed_sobel_image = self.close_sobel_gradient(sobel_image)

        # creating contours and bounding boxes
        contours = self.create_contours(closed_sobel_image)
        bounding_boxes = self.create_bounding_boxes(contours)

        # match the digits
        final_image, output = self.match_digits_in_boxes(grayscale_image, bounding_boxes, digits)

        # print the credit card number
        short_list = ["".join(x) for x in output]
        card_number = " ".join(short_list)
        print(f'Credit Card Number: {card_number}')
        
        # show the final image
        plt.axis('off')
        plt.imshow(final_image, vmin=0, vmax=255)
        plt.show()



    

