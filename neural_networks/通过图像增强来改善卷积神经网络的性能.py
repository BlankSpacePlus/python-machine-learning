from keras.preprocessing.image import ImageDataGenerator

# 创建图像增强对象
augmentation = ImageDataGenerator(featurewise_center=True, zoom_range=0.3, width_shift_range=0.2, horizontal_flip=True,
                                  rotation_range=90)

# 对raw/images文件夹下所有的图像进行处理
augment_images = augmentation.flow_from_directory("raw_images", batch_size=32, class_mode="binary",
                                                  save_to_dir="processed_images")

# 训练神经网络
# network.fit_generator(augment_images, steps_per_epoch=2000, epochs=5, validation_data=augment_images_test,
#                       validation_steps=800)
