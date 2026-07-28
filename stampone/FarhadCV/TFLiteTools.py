"""
Helpers for converting a saved Keras model to TFLite, checking the
resulting model size, and running inference with a TFLite interpreter.
"""

import tensorflow as tf
import tensorflow.lite as lite


#-> lite.TFLiteConverter.from_keras_model_file
#-> lite.TFLiteConverter.from_saved_model()
#-> lite.TFLiteConverter.from_frozen_graph()
#-> lite.TFLiteConverter.from_session()

class TFLite:
    """
    INFO:
    ----------------------------------------------------------
    input:
    ----------------------------------------------------------
    ouput:
    
    """
    def __init__(self):
        self.tfmodel = ''
        self.output_data = 0
    def __help__():
    	pass
    def __version__():
    	pass        
    def Convert(self, model_dir, saveFile):
        #from tensorflow.contrib import lite

        # signature_keys , tags
        self.converter = lite.TFLiteConverter.from_saved_model(model_dir)
        self.tfmodel = self.converter.convert()
        open (saveFile , "wb") .write(self.tfmodel)
        print("model save in {}".format(self.tfmodel))
        
    def Evalute_size_model(self, tfmodel):
        # Show model size in KBs.
        float_model_size = len(tfmodel) / 1024 * 0.001
        print('Float model size = %dMBs.' % float_model_size)
        
    def Optimize(saveFile):
        # Re-convert the model to TF Lite using quantization.
        self.converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_quantized_model = converter.convert()

        # Show model size in KBs. 
        quantized_model_size = (len(tflite_quantized_model) / 1024) * 0.001
        print('Quantized model size = %dKBs,' % quantized_model_size)
        print('which is about %d%% of the float model size.'\
              % (quantized_model_size * 100 / float_model_size))
              
        open (saveFile , "wb") .write(tflite_quantized_model)
        
    def Load_Decoder(tflite_model, image):
        # Load TFLite model and allocate tensors.
        interpreter = tf.lite.Interpreter(model_path = tflite_model)
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Test model on random input data.
        input_shape = input_details[0]['shape']
        #input_data = np.array(image, dtype=np.float32) #np.uint32
        interpreter.set_tensor(input_details[0]['index'], [image])

        interpreter.invoke()

  
        self.output_data = interpreter.get_tensor(output_details[0]['index'])
    
    def Load_Encoder():
        pass
