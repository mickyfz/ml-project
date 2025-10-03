def main():
    print("Hello from ml-project!")

    import torch, tensorflow as tf
    print('--- PyTorch ---')
    print('Version:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('CUDA version:', torch.version.cuda)
        print('GPU name:', torch.cuda.get_device_name(0))

    print('\n--- TensorFlow ---')
    print('Version:', tf.__version__)
    gpus = tf.config.list_physical_devices('GPU')
    print('GPUs available:', len(gpus))
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f'GPU {i}:', gpu.name)



if __name__ == "__main__":
    main()
