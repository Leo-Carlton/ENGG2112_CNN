# Process for model design and hyperparameter choice


# Phase 1, initial choice
* Obviously we will use a CNN for this as it is good for image processing, maintaining local connections
* Resize images to 512x512 as this is a good mid point between image clarity (lost due to downsizing) and processing time (increased by larger dimensions)
* binary cross-entropy loss function as this is a good overall loss function for binary outputs, which ours is
* chose to rank with accuracy, precision, recall, and loss. Recall most important as we care about catching cases, but must be balanced with good precision and then accuracy

# Phase 2, independent testing
* test a bunch of different variables independently to determine rough values that are good for further testing, weed out poor choices
* all tested on 6k images, as good half way point between speed and large amount of images
* base model done first, in cnn_binary_test_batch_fc
* to choose base model, parameters we thought would do well were chosen, based on some limited research
* different optimisers tested, "sgd" and "adam", sgd shows much worse recall so dumped
* batch size tested. 64, 128, and 256 show best results, selected for further testing. Cant test 512 on current model as too large and overflows dimensions of tensor
* fully connected layers tested. 1 and 2 show best results, selected for further testing
* epochs tested. Lower epochs, 5, 10, 15 show best results
* convolutional layers tested. 3 and 5 show best results

# Phase 3, scale up to 40k images
* firstly a test was done with 40k images, to see if results change scaling from 6k to 40k images. Nothing was impacted too much, except epochs
* clearly, as the number of images goes up, more epochs are needed. So a good halfway point of 30 epochs was chosen, this is where it has stabilised and produces confident outputs, before loss goes up too much

# Phase 4, dependent testing
* then, dependent testing was conducted. Every combination of batch size, fully connected layers, and convolutional layers was tested
* batch size 256 showed best results across the board, and was selected
* final selection was batch 256, conv layers 5, fully connected layers 1
* all dependent testing utilised early stopping, as we didnt want it to take too much time to compile a large number of models, and we werent testing epochs

# Phase 5, building model
* with hyperparameters chosen, the final model was built, and showed good results
