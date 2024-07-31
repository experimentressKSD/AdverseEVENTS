import tensorflow as tf 
import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Masking, Bidirectional, TimeDistributed, Concatenate
from tcn import TCN
from keras.layers import *
from keras_nlp.layers import TransformerEncoder, SinePositionEncoding
from tensorflow import linalg, ones


def create_model(input_shape, model):
    if model == 'transformer':
        chosen_model = transformer_model(input_shape)
    elif model == 'TCN':
        chosen_model = online_TCN(input_shape)
    elif model == 'LSTM_enc_dec':
        chosen_model = LSTM_enc_dec(input_shape)
    elif model == 'LSTM':
        chosen_model = online_LSTM(input_shape)

    print("Model name: ", model)

    return chosen_model

def create_padding_mask(input):
    # Create mask which marks the 100000.0 padding values in the input by a 1
    mask = tf.math.not_equal(input, 100000.0) # want 100000.0 to produce 0, aka no attention
    mask = tf.cast(mask, tf.int32)
    mask = mask[:, :, 0]
    return mask
 
def create_lookahead_mask(seq_len):
    # Mask out future entries by marking them with a 0.0
    mask = linalg.band_part(ones((seq_len, seq_len)), -1, 0)
    return mask

def create_full_mask(seq_len):
    mask = ones((seq_len, seq_len))
    return mask
    

def transformer_model(input_shape):

    intermediate_dim = 256
    num_heads=8

    inputs = Input(shape=input_shape)

    pad_mask = create_padding_mask(inputs)

    mask_input = Input(shape=(input_shape[0], input_shape[0]))
    
    embedded_input = TimeDistributed(Dense(64))(inputs)
    pos_encoding = SinePositionEncoding()(embedded_input)
    x = Concatenate()([embedded_input, pos_encoding]) # concatenate the embedded_input and pos_encoding along the time axis

    transformer_block_1 = TransformerEncoder(
        num_heads=num_heads,
        intermediate_dim=intermediate_dim,
        dropout=0.0
    )
    transformer_block_2 = TransformerEncoder(
        num_heads=num_heads,
        intermediate_dim=intermediate_dim,
        dropout=0.0
    )
    transformer_block_3 = TransformerEncoder(
        num_heads=num_heads,
        intermediate_dim=intermediate_dim,
        dropout=0.0
    )


    x = transformer_block_1(x, padding_mask=(pad_mask), attention_mask=(mask_input))
    x = transformer_block_2(x, padding_mask=(pad_mask), attention_mask=(mask_input))
    x = transformer_block_3(x, padding_mask=(pad_mask), attention_mask=(mask_input))
    
    outputs = TimeDistributed(Dense(1, activation='sigmoid'))(x, mask=pad_mask)
    model = tf.keras.Model(inputs=[inputs, mask_input], outputs=outputs)
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], sample_weight_mode='temporal', weighted_metrics=[])
    print(model.summary())
    
    return model


def online_LSTM(input_shape):

    input_layer =  Input(input_shape)
    mask_layer = keras.layers.Masking(mask_value=100000.0)(input_layer)


    lstm1 = (LSTM(500, return_sequences=True))(mask_layer)
    lstm2 = (LSTM(250, return_sequences=True))(lstm1)
    lstm3 = (LSTM(50, return_sequences=True))(lstm2)

    output_layer = TimeDistributed(Dense(1, activation='sigmoid'))(lstm3)

    model = Model(input_layer, output_layer)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, weight_decay=None, epsilon=1e-8)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())

    return model


def online_TCN(input_shape):
    
    input_layer =  Input(input_shape)
    # Masking layer
    mask = (Masking(mask_value=100000.0, input_shape=input_shape))(input_layer)

    pad_mask = create_padding_mask(input_layer)

    # TCN layer
    tcn = (TCN(input_shape=input_shape, nb_filters=128,     
            return_sequences=True, dilations=[1, 2, 4, 8, 16], dropout_rate=0, padding='causal', nb_stacks=2))(mask)

    # Output layer
    output_layer = (TimeDistributed(Dense(1, activation='sigmoid')))(tcn, mask=pad_mask)
   
    model = Model(input_layer, output_layer)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, weight_decay=None, epsilon=1e-8)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    print(model.summary())
    return model


def LSTM_enc_dec(input_shape):

    enc_inputs = Input(shape=input_shape)
    encoder_inputs = keras.layers.Masking(mask_value=100000.0)(enc_inputs)

    encoder = Bidirectional(LSTM(256, return_state=True))
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(encoder_inputs)

    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])
    encoder_states = [state_h, state_c]  # Duplicate the states

    # Set up the decoder, using `encoder_states` as initial state.
    dec_inputs = Input(shape=(None,1))
    decoder_inputs = keras.layers.Masking(mask_value=100000.0)(dec_inputs)
    decoder_lstm = (LSTM(512, return_sequences=True, return_state=True))  # Remains 128
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

    # Add a TimeDistributed Dense layer to make a single prediction for each timestep
    decoder_dense = TimeDistributed(Dense(1, activation='sigmoid'))
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([enc_inputs, dec_inputs], decoder_outputs)

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, weight_decay=None, epsilon=1e-8)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    print(model.summary())
    return model




