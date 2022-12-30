def define_model(configs, gpu, model_type):
    model = None
    model_name = configs.name.lower()

    if model_name == "scunet":
        if model_type == "generator":
            from archs.SCUNet.models import Generator

            model = Generator(configs.generator).to(gpu)

        elif model_type == "discriminator":
            from archs.Discriminators.unet_discriminator import Discriminator

            model = Discriminator(configs.discriminator).to(gpu)

    elif model_name == "realesrgan":
        if model_type == "generator":
            from archs.RealESRGAN.models import Generator

            model = Generator(configs.generator).to(gpu)

        elif model_type == "discriminator":
            from archs.Discriminators.unet_discriminator import Discriminator

            model = Discriminator(configs.discriminator).to(gpu)

    elif model_name == "bsrgan":
        if model_type == "generator":
            from archs.BSRGAN.models import Generator

            model = Generator(configs.generator).to(gpu)

        elif model_type == "discriminator":
            from archs.Discriminators.patchgan_discriminator import (
                Discriminator,
            )

            model = Discriminator(configs.discriminator).to(gpu)

    elif model_name == "edsr":
        if model_type == "generator":
            from archs.EDSR.models import Generator

            model = Generator(configs.generator).to(gpu)

    elif model_name == "swinir":
        if model_type == "generator":
            from archs.SwinIR.models import Generator

            model = Generator(configs.generator).to(gpu)

    assert model != None, "Model is None"

    print(f"Architecture: {model_name} is going to be used")
    return model
