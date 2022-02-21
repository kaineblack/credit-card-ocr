# Extracting Credit Card Numbers from Images 💳
This repository contains a class called `CreditCardOCR` that can be used to extract the 16-digit number from the image of a credit card.<br/><br/>

### Creating Class:
An instance of the class can be instantiated by prodviding an image to analyzye, as well as a path to a reference image for the digits.
```python
credit_card = CreditCardOCR(
    path_to_image='credit_cards/creditcard2.jpeg',
    path_to_reference_image='reference/digit_reference.png'
)
```
<br/><br/>

### Extracting the Credit Card Number:
Then to extract the number, we can use the method `.process_credit_card()`
```python
credit_card.process_credit_card()
```
<br/><br/>

### Sample Output:
<p align="left">
  <img src="https://user-images.githubusercontent.com/72955075/155015061-246047c3-0721-496e-a265-d9dcb1b53525.png" width="500" title="hover text">
</p>
