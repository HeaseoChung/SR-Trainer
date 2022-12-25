def define_model(model, gpu, gan_train):
    generator = None
    discriminator = None
    model_type = model.name.lower()

    if model_type == "scunet":
        from archs.SCUNet.models import Generator

        generator = Generator(model.generator).to(gpu)

        if gan_train:
            from archs.SCUNet.models import Discriminator

            discriminator = Discriminator(model.discriminator).to(gpu)

    elif model_type == "realesrgan":
        from archs.RealESRGAN.models import Generator

        generator = Generator(model.generator).to(gpu)

        if gan_train:
            from archs.RealESRGAN.models import Discriminator

            discriminator = Discriminator(model.discriminator).to(gpu)

    elif model_type == "bsrgan":
        from archs.BSRGAN.models import Generator

        generator = Generator(model.generator).to(gpu)

        from archs.BSRGAN.models import Discriminator

        discriminator = Discriminator(model.discriminator).to(gpu)

    elif model_type == "edsr":
        from archs.EDSR.models import Generator

        generator = Generator(model.generator).to(gpu)

    elif model_type == "swinir":
        from archs.SwinIR.models import Generator

        generator = Generator(model.generator).to(gpu)

    print(f"Architecture: {model_type} is going to be used")
    return generator, discriminator
