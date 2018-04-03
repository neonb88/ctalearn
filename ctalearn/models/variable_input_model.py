import tensorflow as tf

from ctalearn.models.basic import basic_conv_block, basic_head_fc, basic_head_conv
from ctalearn.models.alexnet import (alexnet_block,
        alexnet_head_feature_vector, alexnet_head_feature_map)
from ctalearn.models.mobilenet import mobilenet_block, mobilenet_head
from ctalearn.models.resnet import (resnet_block, resnet_head)
from ctalearn.models.densenet import densenet_block
from ctalearn.models.event_convolution import event_convolution_head

# Drop out all outputs if the telescope was not triggered
def apply_trigger_dropout(inputs,triggers):
    # Reshape triggers from [BATCH_SIZE] to [BATCH_SIZE, WIDTH, HEIGHT, 
    # NUM_CHANNELS]
    triggers = tf.reshape(triggers, [-1, 1, 1, 1])
    triggers = tf.tile(triggers, tf.concat([[1], tf.shape(inputs)[1:]], 0))
    
    return tf.multiply(inputs, triggers)

# Given a list of telescope output features and tensors storing the telescope
# auxiliary parameters (e.g. positions) and trigger list, return a tensor of
# array features of the form [NUM_BATCHES, NUM_TEL, NUM_ARRAY_FEATURES]
def combine_telescopes_as_vectors(telescope_outputs, telescope_aux_inputs, 
        telescope_triggers, is_training):
    array_inputs = []
    combined_telescope_features = []
    combined_telescope_aux_inputs = []
    combined_telescope_triggers = []
    for i, telescope_features in enumerate(telescope_outputs):
        # Flatten output features to get feature vectors
        combined_telescope_features.append(tf.layers.flatten(telescope_features))
        combined_telescope_aux_inputs.append(telescope_aux_inputs[:, i, :])
        combined_telescope_triggers.append(tf.expand_dims(telescope_triggers[:, i], 1))

    # combine telescope features
    combined_telescope_features = tf.stack(combined_telescope_features, axis=1, name="combined_telescope_features")
   
    # aux inputs and telescope triggers are already normalized when loaded
    combined_telescope_aux_inputs = tf.stack(combined_telescope_aux_inputs, axis=1, name="combined_telescope_aux_inputs")
    combined_telescope_triggers = tf.stack(combined_telescope_triggers, axis=1, name="combined_telescope_triggers") 

    # Insert auxiliary input into each feature vector
    array_features = tf.concat([combined_telescope_features,combined_telescope_aux_inputs,combined_telescope_triggers], axis=2, name="combined_array_features") 
   
    return array_features

# Given a list of telescope output features and tensors storing the telescope
# positions and trigger list, return a tensor of array features of the form
# [NUM_BATCHES, TEL_OUTPUT_WIDTH, TEL_OUTPUT_HEIGHT, (TEL_OUTPUT_CHANNELS + 
#       NUM_AUXILIARY_INPUTS_PER_TELESCOPE) * NUM_TELESCOPES]
def combine_telescopes_as_feature_maps(telescope_outputs, telescope_aux_inputs, 
        telescope_triggers, is_training):
    array_inputs = []
    for i, telescope_features in enumerate(telescope_outputs):
        # Get the telescope auxiliary parameters (e.g. position)
        # [NUM_BATCH, NUM_AUX_PARAMS]
        telescope_aux_input = telescope_aux_inputs[:, i, :]
        # Get whether the telescope triggered [NUM_BATCH]
        telescope_trigger = telescope_triggers[:, i]
        # Tile the aux params along the width and height dimensions
        telescope_aux_input = tf.expand_dims(telescope_aux_input, 1)
        telescope_aux_input = tf.expand_dims(telescope_aux_input, 1)
        telescope_aux_input = tf.tile(telescope_aux_input,
                tf.concat([[1], tf.shape(telescope_features)[1:-1], [1]], 0))
        # Tile the trigger along the width, height, and channel dimensions
        telescope_trigger = tf.reshape(telescope_trigger, [-1, 1, 1, 1])
        telescope_trigger = tf.tile(telescope_trigger,
                tf.concat([[1], tf.shape(telescope_features)[1:-1], [1]], 0))
        # Insert auxiliary input as additional channels in feature maps
        telescope_features = tf.concat([telescope_features, 
            telescope_aux_input, telescope_trigger], 3)
        array_inputs.append(telescope_features)
    array_features = tf.concat(array_inputs, axis=3)

    return array_features

# Given a list of telescope output features and tensors storing the telescope
# positions and trigger list, return a tensor of array features of the form
# [BATCH_SIZE, NUM_TELS, TEL_OUTPUT_CHANNELS] and a tensor of the form
# [BATCH_SIZE, NUM_AUXILIARY_INPUTS_PER_TELESCOPE * NUM_TELESCOPES]
def combine_telescopes_as_event(telescope_outputs, telescope_aux_inputs, 
        telescope_triggers, is_training):

    # Combine telescope outputs as an event [BATCH_SIZE, N_TEL, M_FEATURES]
    telescope_data = [tf.expand_dims(tf.layers.flatten(tel_out), 1) for
            tel_out in telescope_outputs]
    event_data = tf.concat(telescope_data, 1)

    # Combine telescope auxiliary inputs and triggers into a single vector
    auxiliary_data = tf.concat([tf.layers.flatten(telescope_aux_inputs),
        tf.layers.flatten(telescope_triggers)], 1)

    return [event_data, auxiliary_data]

def variable_input_model(features, labels, params, is_training):
   
    # Reshape inputs into proper dimensions
    num_telescope_types = len(params['processed_telescope_types']) 
    if not num_telescope_types == 1:
        raise ValueError('Must use a single telescope type for Variable Input Model. Number used: {}'.format(num_telescope_types))
    telescope_type = params['processed_telescope_types'][0]
    image_width, image_length, image_depth = params['processed_image_shapes'][telescope_type]
    num_telescopes = params['processed_num_telescopes'][telescope_type]
    num_aux_inputs = sum(params['processed_aux_input_nums'].values())
    num_gamma_hadron_classes = params['num_classes']
    
    telescope_data = features['telescope_data']
    telescope_data = tf.reshape(telescope_data, [-1, num_telescopes, 
        image_width, image_length, image_depth], name="telescope_images")
    
    telescope_triggers = features['telescope_triggers']
    telescope_triggers = tf.reshape(telescope_triggers, [-1, num_telescopes])
    telescope_triggers = tf.cast(telescope_triggers, tf.float32, name="telescope_triggers")

    telescope_aux_inputs = features['telescope_aux_inputs']
    telescope_aux_inputs = tf.reshape(telescope_aux_inputs,
            [-1, num_telescopes, num_aux_inputs], name="telescope_aux_inputs")
    
    # Reshape labels to vector as expected by tf.one_hot
    gamma_hadron_labels = labels['gamma_hadron_label']
    gamma_hadron_labels = tf.reshape(gamma_hadron_labels, [-1])

    # Split data by telescope by switching the batch and telescope dimensions
    # leaving width, length, and channel depth unchanged
    telescope_data = tf.transpose(telescope_data, perm=[1, 0, 2, 3, 4])

    # Define the network being used. Each CNN block analyzes a single
    # telescope. The outputs for non-triggering telescopes are zeroed out 
    # (effectively, those channels are dropped out).
    # Unlike standard dropout, this zeroing-out procedure is performed both at
    # training and test time since it encodes meaningful aspects of the data.
    # The telescope outputs are then stacked into input for the array-level
    # network, either into 1D feature vectors or into 3D convolutional 
    # feature maps, depending on the requirements of the network head.
    # The array-level processing is then performed by the network head. The
    # logits are returned and fed into a classifier.

    # Choose the CNN block
    if params['cnn_block'] == 'alexnet':
        cnn_block = alexnet_block
    elif params['cnn_block'] == 'mobilenet':
        cnn_block = mobilenet_block
    elif params['cnn_block'] == 'resnet':
        cnn_block = resnet_block
    elif params['cnn_block'] == 'densenet':
        cnn_block = densenet_block
    elif params['cnn_block'] == 'basic':
        cnn_block = basic_conv_block
    else:
        raise ValueError("Invalid CNN block specified: {}.".format(params['cnn_block']))

    # Choose the network head and telescope combination method
    if params['network_head'] == 'alexnet_fc':
        network_head = alexnet_head_feature_vector
        combine_telescopes = combine_telescopes_as_vectors
    elif params['network_head'] == 'alexnet_conv':
        network_head = alexnet_head_feature_map
        combine_telescopes = combine_telescopes_as_feature_maps
    elif params['network_head'] == 'mobilenet':
        network_head = mobilenet_head
        combine_telescopes = combine_telescopes_as_feature_maps
    elif params['network_head'] == 'resnet':
        network_head = resnet_head
        combine_telescopes = combine_telescopes_as_feature_maps
    elif params['network_head'] == 'basic_fc':
        network_head = basic_head_fc
        combine_telescopes = combine_telescopes_as_vectors
    elif params['network_head'] == 'basic_conv':
        network_head = basic_head_conv
        combine_telescopes = combine_telescopes_as_feature_maps
    elif params['network_head'] == 'eventconvolution':
        network_head = event_convolution_head
        combine_telescopes = combine_telescopes_as_event
    else:
        raise ValueError("Invalid network head specified: {}.".format(params['network_head']))
    
    # Process the input for each telescope
    telescope_outputs = []
    for telescope_index in range(num_telescopes):
        # Set all telescopes after the first to share weights
        if telescope_index == 0:
            reuse = None
        else:
            reuse = True
        with tf.variable_scope("CNN_block", reuse=reuse):
            telescope_features = cnn_block(
                tf.gather(telescope_data, telescope_index), 
                params=params,
                is_training=is_training,
                reuse=reuse)

        if params['pretrained_weights']:
            tf.contrib.framework.init_from_checkpoint(params['pretrained_weights'],{'CNN_block/':'CNN_block/'})

        telescope_features = apply_trigger_dropout(telescope_features,
                tf.gather(telescope_triggers, telescope_index, axis=1))
        telescope_outputs.append(telescope_features)

    # Process the single telescope data into array-level input
    array_features = combine_telescopes(
            telescope_outputs, 
            telescope_aux_inputs, 
            telescope_triggers,
            is_training)
   
    with tf.variable_scope("NetworkHead"):
        # Process the combined array features
        logits = network_head(array_features, params=params,
                is_training=is_training)

    return logits
