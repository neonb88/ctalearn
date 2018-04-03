import tensorflow as tf

AUXILIARY_VECTOR_LAYERS = [512, 32]

EVENT_CONVOLUTION_LAYERS = [(32, 32), (32, 32), (32, 32), (32, 32)]

def event_convolution_head(array_features, params=None, is_training=True):
    input_layer = array_features[0]
    auxiliary_input = array_features[1]

    # Get hyperparameters
    if params is None: params = {}
    num_classes = params.get('num_classes', 2)
    
    # Process auxiliary input into a vector representation
    aux_vector = tf.layers.flatten(auxiliary_input)
    for i, num_units in enumerate(AUXILIARY_VECTOR_LAYERS):
        aux_vector = tf.layers.dense(aux_vector, units=num_units,
                activation=tf.nn.relu,
                name="aux_input_processing_{}".format(i+1))
    # Tile to [BATCH_SIZE, D_AUX]
    aux_vector = tf.tile(tf.expand_dims(aux_vector, 0),
            tf.shape(input_layer)[0])

    x = input_layer # [BATCH_SIZE, N_TEL, M_CHANNEL]
    # Apply event convolutions
    for i, (num_outputs, num_filters) in enumerate(EVENT_CONVOLUTION_LAYERS):
        # Tile auxiliary vector to [BATCH_SIZE, D_AUX, M_CHANNEL]
        aux_vectors = tf.tile(tf.expand_dims(aux_vector, -1), tf.shape(x)[-1])
        # Concatenate auxiliary vector to the outputs along the width
        # dimension: [BATCH_SIZE, N_TEL + D_AUX, M_CHANNEL]
        tf.concat([x, aux_vectors], 1)
        # Perform separable convolution: first apply N_OUT filters depthwise
        # (acting separately on channels), then M_OUT filters pointwise mixing
        # channels: [BATCH_SIZE, N_OUTPUT, M_FILTERS]
        x = tf.separable_conv1d(x, num_filters, tf.shape(x)[1],
                depth_multiplier=num_outputs,
                name="event_convolution_{}".format(i+1))

    # Get output logits
    flatten = tf.layers.flatten(x)
    logits = tf.layers.dense(flatten, units=num_classes, name="logits")

    return logits
