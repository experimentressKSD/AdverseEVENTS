import tensorflow as tf
from keras.layers import Input, Conv1D, GlobalAveragePooling1D, Dense, Attention, BatchNormalization, ReLU, LSTM, Dropout, Conv1DTranspose, Flatten, TimeDistributed, Concatenate
from keras.models import Model, Sequential
from tcn import TCN
import keras
from functools import partial

"""

# CONV1D WITH ATTENTION
def create_conv1d_attention_model(input_shape):
    inputs = Input(shape=input_shape)

    x = Conv1D(64, kernel_size=3, activation='relu')(inputs)
    x = Conv1D(128, kernel_size=3, activation='relu')(x)
    attention = Attention()([x, x])
    x = GlobalAveragePooling1D()(attention)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(input_shape[0], activation='sigmoid')(x)  # Sigmoid for binary classification

    model = Model(inputs=inputs, outputs=outputs)

    return model

"""

def create_model(input_shape, target_shape, model):
    if model == 'baseline':
        chosen_model = create_baseline(input_shape, target_shape)
    elif model == '1DConv':
        chosen_model = create_conv1d_model(input_shape, target_shape)
    elif model == 'TCN':
        chosen_model = create_TCN_model(input_shape, target_shape)
    elif model == 'LSTM_enc_dec':
        chosen_model = create_lstm_encoder_decoder(input_shape, target_shape)
    elif model == 'LSTM_attn':
        chosen_model = create_lstm_encoder_decoder_attn(input_shape, target_shape)
    elif model == 'LSTM':
        chosen_model = create_lstm_model(input_shape, target_shape)

    print("Model name: ", model)

    print(chosen_model.summary())

    return chosen_model

def learning_rate_schedule(epoch, lr, step_lr_epoch_div, step_lr_div_factor):
    if epoch%step_lr_epoch_div == 0 and epoch != 0:
        return lr * step_lr_div_factor
    else:
        return lr

def get_compiled_model(model, lr, step_lr_epoch_div, step_lr_div_factor):

    # lr = 0.001
    # loss = tf.keras.losses.BinaryCrossentropy(from_logits=False) # from_logits=False because we use a sigmoid in the last layer of each model

    # optimizer = tf.keras.optimizers.Adam(learning_rate=lr, weight_decay=None, epsilon=1e-8)
    # # create learning rate scheduler if needed

    # model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, weight_decay=None, epsilon=1e-8)

    lr_schedule = partial(learning_rate_schedule, step_lr_epoch_div=step_lr_epoch_div, step_lr_div_factor=step_lr_div_factor)
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

    model.compile(loss=loss_object, optimizer=optimizer, metrics=['accuracy'])

    return model, lr_scheduler

def create_lstm_encoder_decoder(input_shape, target_shape):
    # Define an input sequence and process it.

    latent_dim = 256
    num_decoder_tokens = target_shape[0]

    # Define an input sequence and process it.
    encoder_inputs = Input(shape=input_shape)
    encoder = LSTM(64, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)

    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=target_shape)
    decoder_lstm = LSTM(64, return_sequences=True)
    decoder_outputs = decoder_lstm(decoder_inputs, initial_state=encoder_states)

    # Add a TimeDistributed Dense layer to make a prediction for each timestep
    decoder_dense = TimeDistributed(Dense(1, activation='sigmoid'))
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model

def create_lstm_encoder_decoder_attn(input_shape, target_shape):

    # Define an input sequence and process it.
    encoder_inputs = Input(shape=input_shape)
    encoder = LSTM(64, return_sequences=True, return_state=True, activation='tanh')
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)

    # We keep `encoder_outputs` for attention and states for decoder initial state.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=target_shape)
    decoder_lstm = LSTM(64, return_sequences=True, activation='tanh')
    decoder_outputs = decoder_lstm(decoder_inputs, initial_state=encoder_states)

    # Attention layer
    attention = Attention()
    attention_outputs = attention([decoder_outputs, encoder_outputs])

    # Concat attention output and decoder LSTM output
    decoder_concat_input = Concatenate(axis=-1)([decoder_outputs, attention_outputs])

    # Add a TimeDistributed Dense layer to make a prediction for each timestep
    decoder_dense = TimeDistributed(Dense(1, activation='sigmoid'))
    decoder_outputs = decoder_dense(decoder_concat_input)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return model


# CONV1D WITH ATTENTION
def create_conv1d_attention_model(input_shape, target_shape):
    
    input_layer =  Input(input_shape)

    conv1 = Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = ReLU()(conv1)
    attention1 = Attention()([conv1, conv1])

    conv2 = Conv1D(filters=64, kernel_size=3, padding="same")(attention1)
    conv2 = BatchNormalization()(conv2)
    conv2 = ReLU()(conv2)
    attention2 = Attention()([conv2, conv2])

    conv3 = Conv1D(filters=64, kernel_size=3, padding="same")(attention2)
    conv3 = BatchNormalization()(conv3)
    conv3 = ReLU()(conv3)
    attention3 = Attention()([conv3, conv3])

    gap = GlobalAveragePooling1D()(attention3)

    output_layer = Dense(target_shape[0], activation="sigmoid")(gap)

    return Model(inputs=input_layer, outputs=output_layer)


# CONV1D WITHOUT ATTENTION
def create_conv1d_model_bn(input_shape, target_shape):
    
    input_layer =  Input(input_shape)

    conv1 = Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = ReLU()(conv1)

    conv2 = Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = ReLU()(conv2)

    conv3 = Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = ReLU()(conv3)

    gap = GlobalAveragePooling1D()(conv3)

    output_layer = Dense(target_shape[0], activation="sigmoid")(gap)

    return Model(inputs=input_layer, outputs=output_layer)

def create_conv1d_model(input_shape, target_shape):
    input_layer = Input(input_shape)

    conv1 = Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = ReLU()(conv1)

    conv2 = Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = ReLU()(conv2)

    conv3 = Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = ReLU()(conv3)

    gap = GlobalAveragePooling1D()(conv3)

    output_layer = Dense(target_shape[0], activation="sigmoid")(gap)

    return Model(inputs=input_layer, outputs=output_layer)

def create_baseline(input_shape, target_shape):
        
    model_conv = Sequential(
        [
            Input(shape=input_shape),
            Conv1D(
                filters=64, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            Dropout(rate=0.2),
            Conv1D(
                filters=64, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            Conv1D(
                filters=64, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            Dropout(rate=0.2),
            Conv1D(
                filters=64, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            Conv1DTranspose(
                filters=64, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            Dropout(rate=0.2),
            Conv1DTranspose(
                filters=64, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            Conv1DTranspose(filters=7, kernel_size=7, padding="same"),
            Conv1DTranspose(filters=7, kernel_size=7, padding="same"),
            Flatten(),
            Dense(target_shape[0], activation="sigmoid")
            
        ]
    )

    return model_conv

# LSTM with attention
def create_lstm_model_old(input_shape, target_shape):

    input_layer =  Input(input_shape)

    lstm1 = LSTM(100, return_sequences=True)(input_layer)
    lstm2 = LSTM(50, return_sequences=False)(lstm1)

    dense1 = Dense(50, activation='relu')(lstm2)

    output_layer = Dense(target_shape[0], activation="sigmoid")(dense1)

    return Model(inputs=input_layer, outputs=output_layer)


def create_lstm_model(input_shape, target_shape):

    input_layer =  Input(input_shape)

    lstm1 = LSTM(100, return_sequences=True)(input_layer)
    lstm2 = LSTM(50, return_sequences=True)(lstm1)
    lstm3 = LSTM(25, return_sequences=True)(lstm2)
    output_layer = TimeDistributed(Dense(1, activation='sigmoid'))(lstm3)

    return Model(inputs=input_layer, outputs=output_layer)


# TCN (conv + dilation)
def create_TCN_model(input_shape, target_shape):
    model = Sequential([
        TCN(input_shape=input_shape, nb_filters=64,     
            return_sequences=True, dilations=[1, 2, 4, 8, 16], dropout_rate=0),
        TimeDistributed(Dense(32, activation='relu')),
        TimeDistributed(Dense(16, activation='relu')),
        TimeDistributed(Dense(1, activation='sigmoid'))
    ])

    return model



def create_autoencoder(input_shape):

    model = Sequential(
        [
            Input(shape=input_shape),
            Conv1D(
                filters=64, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            Dropout(rate=0.2),
            Conv1D(
                filters=64, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            Conv1DTranspose(
                filters=64, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            Dropout(rate=0.2),
            Conv1DTranspose(
                filters=64, kernel_size=5, padding="same", strides=2, activation="relu"
            ),
            Conv1DTranspose(filters=7, kernel_size=5, padding="same"),
        ]
    )

    return model


