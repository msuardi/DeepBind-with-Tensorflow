import tensorflow as tf
import numpy as np
import math
import random
from keras import backend as K
from keras.layers import Concatenate


dictpad={'A':[1.,0.,0.,0.],'C':[0.,1.,0.,0.],'G':[0.,0.,1.,0.],'T':[0.,0.,0.,1.],'U':[0.,0.,0.,1.],'N':[0.25,0.25,0.25,0.25]}
dictReverse={'A':'T','C':'G','G':'C','T':'A','N':'N'} #dictionary to implement reverse-complement mode

#metodo per effettuare la concatenazione alternata, modificato il concatenate originale
def altconcatenate(tensors,axis=-1):
    """Concatenates a list of tensors alongside the specified axis.
    # Arguments
        tensors: list of tensors to concatenate.
        axis: concatenation axis.
    # Returns
        A tensor.
    """
    if axis < 0:
        rank = K.ndim(tensors[0])
        if rank:
            axis %= rank
        else:
            axis = 0

    if K.py_all([K.is_sparse(x) for x in tensors]):
        return tf.sparse_concat(axis, tensors)
    else:
        result=tf.concat([K.to_dense(x) for x in tensors], 2)
        result=tf.reshape(result,[-1,32])
        result=tf.expand_dims(result,2)
        return result

#definisco la classe AltConcatenate ereditando dalla originale Concatenate
class AltConcatenate(Concatenate):
    """Layer that concatenates a list of inputs.
    It takes as input a list of tensors,
    all of the same shape expect for the concatenation axis,
    and returns a single tensor, the concatenation of all inputs.
    # Arguments
        axis: Axis along which to concatenate.
        **kwargs: standard layer keyword arguments.
    """

    def __init__(self, axis=-1, **kwargs):
        super(Concatenate, self).__init__(**kwargs)
        self.axis = axis
        self.supports_masking = True

    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape, list):
            raise ValueError('`Concatenate` layer should be called '
                             'on a list of inputs')
        if all([shape is None for shape in input_shape]):
            return
        reduced_inputs_shapes = [list(shape) for shape in input_shape]
        shape_set = set()
        for i in range(len(reduced_inputs_shapes)):
            del reduced_inputs_shapes[i][self.axis]
            shape_set.add(tuple(reduced_inputs_shapes[i]))
        if len(shape_set) > 1:
            raise ValueError('`Concatenate` layer requires '
                             'inputs with matching shapes '
                             'except for the concat axis. '
                             'Got inputs shapes: %s' % (input_shape))

    def call(self, inputs):
        if not isinstance(inputs, list):
            raise ValueError('A `Concatenate` layer should be called '
                             'on a list of inputs.')
        return altconcatenate(inputs, axis=2)

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('A `Concatenate` layer should be called '
                             'on a list of inputs.')
        input_shapes = input_shape
        output_shape = list(input_shapes[0])
        for shape in input_shapes[1:]:
            if output_shape[self.axis] is None or shape[self.axis] is None:
                output_shape[self.axis] = None
                break
            output_shape[self.axis] += shape[self.axis]
        return tuple(output_shape)

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        if not isinstance(mask, list):
            raise ValueError('`mask` should be a list.')
        if not isinstance(inputs, list):
            raise ValueError('`inputs` should be a list.')
        if len(mask) != len(inputs):
            raise ValueError('The lists `inputs` and `mask` '
                             'should have the same length.')
        if all([m is None for m in mask]):
            return None
        # Make a list of masks while making sure
        # the dimensionality of each mask
        # is the same as the corresponding input.
        masks = []
        for input_i, mask_i in zip(inputs, mask):
            if mask_i is None:
                # Input is unmasked. Append all 1s to masks,
                # but cast it to bool first
                masks.append(K.cast(K.ones_like(input_i), 'bool'))
            elif K.ndim(mask_i) < K.ndim(input_i):
                # Mask is smaller than the input, expand it
                masks.append(K.expand_dims(mask_i))
            else:
                masks.append(mask_i)
        concatenated = altconcatenate(masks, axis=self.axis)
        return K.all(concatenated, axis=-1, keepdims=False)

    def get_config(self):
        config = {
            'axis': self.axis,
        }
        base_config = super(Concatenate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




#funzione per trasformare le sequenze di input in array, presente nel paper supplementare pag.3
def seqtopad(sequence,motlen):
    pad=[]
    for j in range(motlen-1):
        pad.extend(dictpad['N'])
    res=pad.copy()
    for i in range(len(sequence)):
        res.extend(dictpad[sequence[i]])
    res.extend(pad)
    return np.asarray(res)

#funzione per trovare filamento opposto della sequenza
def reverse(sequence):
    revseq=''
    for i in sequence:
        revseq+=dictReverse[i]
    return revseq    

#funzioni per inizializzare momentum_rate e learning_rate, paper suppl. pag. 12
def logsampler(a,b,tensor=0):
    if(tensor==1):
        x=tf.Variable(tf.random_uniform([1],minval=0,maxval=1))
    else:
        x=np.random.uniform(low=0,high=1)
    y=10**((math.log10(b)-math.log10(a))*x + math.log10(a))
    return y

def sqrtsampler(a,b,tensor=0):
    if(tensor==1):
        x=tf.Variable(tf.random_uniform([1],minval=0,maxval=1))
    else:
        x=np.random.uniform(low=0,high=1)
    y=(b-a)*math.sqrt(x)+a
    return y

#funzione per effettuare il padding delle sequenze nel caso in cui gli input hanno sequenze di lunghezza diversa
#necessario per RNAcompete, anche perché i tensori accettano input della stessa forma
def padsequence(sequence,maxlength):
    return sequence + 'N'*(maxlength-len(sequence))
    
#funzione di perdita per CHIP-seq e SELEX, non è di default in Kears
#TODO va in loop 
def log_loss(label,prediction):
    def sigma(x):
        return 1/(1+math.e**(-x))
    return K.mean(-label*K.log(sigma(prediction)) - (1-label)*K.log(1-sigma(prediction)))

#funzione per creare le sequenze di training con specificità 0 nel caso di CHIP e SELEX (forniti solo quelli "positivi")
def dinucshuffle(sequence):
    b=[sequence[i:i+2] for i in range(0, len(sequence), 2)]
    random.shuffle(b)
    d=''.join([str(x) for x in b])
    return d
    
