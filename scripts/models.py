def define_model(model, gpu, gan_train):
    generator = None
    discriminator = None

    if model.name.lower() == "scunet":
        from archs.SCUNet.models import Generator

        generator = Generator(model.generator).to(gpu)

        if gan_train:
            from archs.SCUNet.models import Discriminator

            discriminator = Discriminator(model.discriminator).to(gpu)

    elif model.name.lower() == "realesrgan":
        from archs.RealESRGAN.models import Generator

        generator = Generator(model.generator).to(gpu)

        if gan_train:
            from archs.RealESRGAN.models import Discriminator

            discriminator = Discriminator(model.discriminator).to(gpu)

    elif model.name.lower() == "bsrgan":
        from archs.BSRGAN.models import Generator

        generator = Generator(model.generator).to(gpu)

        from archs.BSRGAN.models import Discriminator

        discriminator = Discriminator(model.discriminator).to(gpu)

    elif model.name.lower() == "edsr":
        from archs.EDSR.models import Generator

        generator = Generator(model.generator).to(gpu)

    elif model.name.lower() == "swinir":
        from archs.SwinIR.models import Generator

        generator = Generator(model.generator).to(gpu)

    return generator, discriminator
