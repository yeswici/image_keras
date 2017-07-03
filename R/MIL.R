#1 - Loading Tensorflow and Keras
#library(devtools)
#devtools::install_github("rstudio/tensorflow")
devtools::install_github("rstudio/reticulate",force = TRUE)

library(tensorflow)
install_tensorflow()  
#install_tensorflow(gpu = TRUE)  
devtools::install_github("rstudio/keras")
library(keras)

#2 - File locations and settings
#setwd("/home/ubuntu")
train_directory <- "image_keras/data/train"
validation_directory <- "image_keras/data/validation"

img_width <- 150
img_height <- 150
batch_size <- 32
epochs <- 30
train_samples = 2048
validation_samples = 832

#3 - Loading images
train_generator <- flow_images_from_directory(train_directory, generator = image_data_generator(),
                                              target_size = c(img_width, img_height), color_mode = "rgb",
                                              class_mode = "binary", batch_size = batch_size, shuffle = TRUE,
                                              seed = 123)

validation_generator <- flow_images_from_directory(validation_directory, generator = image_data_generator(),
                                                   target_size = c(img_width, img_height), color_mode = "rgb", classes = NULL,
                                                   class_mode = "binary", batch_size = batch_size, shuffle = TRUE,
                                                   seed = 123)

#4 - Small Conv Net - 1) Model architecture definition
model <- keras_model_sequential()

model %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3), input_shape = c(img_width, img_height, 3)) %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  
  layer_conv_2d(filter = 32, kernel_size = c(3,3)) %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  
  layer_conv_2d(filter = 64, kernel_size = c(3,3)) %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  
  layer_flatten() %>%
  layer_dense(64) %>%
  layer_activation("relu") %>%
  layer_dropout(0.5) %>%
  layer_dense(1) %>%
  layer_activation("sigmoid")

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 0.0001, decay = 1e-6),
  metrics = "accuracy"
)

#4 - Small Conv Net - 2) training
model %>% fit_generator(
  train_generator,
  steps_per_epoch = as.integer(train_samples/batch_size), 
  epochs = epochs, 
  validation_data = validation_generator,
  validation_steps = as.integer(validation_samples/batch_size),
  verbose=2
)

#system("sudo -H pip install h5py")
#save_model_weights_hdf5(model, 'models/basic_cnn_30_epochsR.h5', overwrite = TRUE)

#4 - Small Conv Net - 3)Evaluating on validation set
evaluate_generator(model,validation_generator, validation_samples)

#5 - Data augmentation for improving the model
# Defining the model ------------------------------------------------------
#img_width <- 150
#img_height <- 150

model <- keras_model_sequential()

model %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3), input_shape = c(img_width, img_height, 3)) %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  
  layer_conv_2d(filter = 32, kernel_size = c(3,3)) %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  
  layer_conv_2d(filter = 64, kernel_size = c(3,3)) %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  
  layer_flatten() %>%
  layer_dense(64) %>%
  layer_activation("relu") %>%
  layer_dropout(0.5) %>%
  layer_dense(1) %>%
  layer_activation("sigmoid")

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 0.001, decay = 1e-6),
  metrics = "accuracy"
)

augment <- image_data_generator(rescale=1./255,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=TRUE)

train_generator_augmented <- flow_images_from_directory(train_directory, generator = augment,
                                                        target_size = c(img_width, img_height), color_mode = "rgb",
                                                        class_mode = "binary", batch_size = batch_size, shuffle = TRUE,
                                                        seed = 123)

validation_generator <- flow_images_from_directory(validation_directory, generator = image_data_generator(rescale=1./255),
                                                   target_size = c(img_width, img_height), color_mode = "rgb", classes = NULL,
                                                   class_mode = "binary", batch_size = batch_size, shuffle = TRUE,
                                                   seed = 123)

model %>% fit_generator(
  train_generator_augmented,
  steps_per_epoch = as.integer(train_samples/batch_size), 
  epochs = epochs, 
  validation_data = validation_generator,
  validation_steps = as.integer(validation_samples/batch_size),
  verbose=2
)

save_model_hdf5(model, 'models/augmented_30_epochsR.h5', overwrite = TRUE)

#Evaluating on validation set
evaluate_generator(model,validation_generator, validation_samples)

#6 - Using a pre-trained model
#VGG16 model is available in Keras

model_vgg <- application_vgg16(include_top = FALSE, weights = "imagenet")

#Using the VGG16 model to process samples

train_generator_bottleneck <- flow_images_from_directory(
  train_directory,
  target_size= c(img_width, img_height),
  batch_size=batch_size,
  class_mode=NULL,
  shuffle=FALSE)

validation_generator_bottleneck <- flow_images_from_directory(
  validation_directory,
  target_size= c(img_width, img_height),
  batch_size=batch_size,
  class_mode=NULL,
  shuffle=FALSE)

# This is a long process, so we save the output of the VGG16 once and for all.

bottleneck_features_train <- predict_generator(model_vgg,train_generator_bottleneck, train_samples / batch_size)
saveRDS(bottleneck_features_train, "models/bottleneck_features_train.rds")
bottleneck_features_validation <- predict_generator(model_vgg,validation_generator_bottleneck, validation_samples / batch_size)
saveRDS(bottleneck_features_validation, "models/bottleneck_features_validation.rds")

bottleneck_features_train <- readRDS("models/bottleneck_features_train.rds")
bottleneck_features_validation <- readRDS("models/bottleneck_features_validation.rds")
train_labels = c(rep(0,train_samples/2),rep(1,train_samples/2))
validation_labels = c(rep(0,validation_samples/2),rep(1,validation_samples/2))

#7 - define and train the custom fully connected neural network :
model_top <- keras_model_sequential()

model_top %>%
  layer_dense(units=nrow(bottleneck_features_train),input_shape = dim(bottleneck_features_train)[2:4]) %>% 
  layer_flatten() %>%
  layer_dense(256) %>%
  layer_activation("relu") %>%
  layer_dropout(0.5) %>%
  layer_dense(1) %>%
  layer_activation("sigmoid")

model_top %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 0.0001, decay = 1e-6),
  metrics = "accuracy")
valid = list(bottleneck_features_validation, validation_labels)
model_top %>% fit(
  x = bottleneck_features_train, y = train_labels,
  epochs=epochs, 
  batch_size=16,  ##Hit out of memory with a batch size of 32
  validation_data=valid,
  verbose=2)

save_model_weights_hdf5(model_top, 'models/bottleneck_30_epochsR.h5', overwrite = TRUE)

evaluate(model_top,bottleneck_features_validation, validation_labels)

#8 - Fine-tuning the top layers of a a pre-trained network
model_vgg <- application_vgg16(include_top = FALSE, weights = "imagenet",input_shape =c (img_width, img_height, 3))

top_model <- keras_model_sequential()

top_model %>%
  layer_dense(units=nrow(bottleneck_features_train),input_shape = model_vgg$output_shape[2:4]) %>% 
  layer_flatten() %>%
  layer_dense(256) %>%
  layer_activation("relu") %>%
  layer_dropout(0.5) %>%
  layer_dense(1) %>%
  layer_activation("sigmoid")

load_model_weights_hdf5(top_model, "models/bottleneck_30_epochsR.h5")

model_ft <- keras_model(inputs = model_vgg$input, outputs = top_model(model_vgg$output))

for (layer in model_ft$layers[1:16])
  layer$trainable <- FALSE

model_ft %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_sgd(lr=1e-3, momentum=0.9),
  metrics = "accuracy")

augment <- image_data_generator(rescale=1./255,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=TRUE)

train_generator_augmented <- flow_images_from_directory(train_directory, generator = augment,
                                                        target_size = c(img_width, img_height), color_mode = "rgb",
                                                        class_mode = "binary", batch_size = batch_size, shuffle = TRUE,
                                                        seed = 123)

validation_generator <- flow_images_from_directory(validation_directory, generator = image_data_generator(rescale=1./255),
                                                   target_size = c(img_width, img_height), color_mode = "rgb", classes = NULL,
                                                   class_mode = "binary", batch_size = batch_size, shuffle = TRUE,
                                                   seed = 123)


model_ft %>% fit_generator(
  train_generator_augmented,
  steps_per_epoch = as.integer(train_samples/batch_size), 
  epochs = epochs, 
  validation_data = validation_generator,
  validation_steps = as.integer(validation_samples/batch_size),
  verbose=2
)

save_model_weights_hdf5(model_ft, 'finetuning_30epochs_vggR.h5', overwrite = TRUE)
load_model_weights_hdf5(model_ft, 'finetuning_30epochs_vggR.h5')

#Evaluating on validation set
evaluate_generator(model_ft,validation_generator, validation_samples)

