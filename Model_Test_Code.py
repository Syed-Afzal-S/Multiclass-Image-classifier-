
import tensorflow as tf
import numpy as np

# Loading the saved model
model = tf.keras.models.load_model('Face_Recog_Model')

preds = np.round(model.predict(test_data),0) # rounding off prediction value to a integer
print('rounded_test_labels', preds)

# Output will be in the form of array [0,0,0,0,0,0]
# 0th index - Cats, 1st index - dogs, 2nd index - messi, 3rd index - salman, 4th-index - sharukh, 5th index - syed
