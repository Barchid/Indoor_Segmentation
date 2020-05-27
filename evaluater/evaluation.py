from evaluater.quantitative import evaluate_accuracy, evaluate_speed, evaluate_acc
from evaluater.qualitative import visualize_results
from utils import factory
from data_generators.segmentation_data_generator import SegmentationDataGenerator


def load_weights(model, config):
    """Load weights file if required by configuration
    """
    if not hasattr(config, 'validation') or type(config.validation.weights_file) != str:
        return

    print('Load weight file : ', config.validation.weights_file)
    model.load_weights(config.validation.weights_file)


def quantitative(model, config, datagen):
    """Quantitative evaluation 
    """
    print('################# PROCESSING METRICS FOR ACCURACY ###############')
    Aiou, class_Aiou, pixel_acc, mean_pixel_acc = evaluate_accuracy(
        model, config, datagen)
    # evaluate_acc(model, config, datagen)

    print('################# PROCESSING METRICS FOR INFERENCE TIME ###############')
    # total_time, fps = evaluate_speed(model, config, datagen)

    print('################# Results : ##################')
    print('Pixel accuracy : ', pixel_acc)
    print('Mean pixel accuracy : ', mean_pixel_acc)
    print('Mean IoU : ', Aiou)
    print('Per-class mIoUs : ', class_Aiou)
    # print('Total inference time :', total_time)
    # print('FPS :', fps)


def qualitative(model, config, datagen):
    """Qualitative evaluation (visualize all the results)
    """
    print('################ BEGINNING VISUALIZATION ############')
    print('--- Press A to stop visualization')
    print('--- Press any other key to continue')
    visualize_results(model, config, datagen)


def evaluation(config, visualization=False):
    # Data generator creation
    datagen = SegmentationDataGenerator(config, is_training_set=False)

    # dynamic model instanciation
    network = None
    if config.model.class_name == 'models.fpn_net.FpnNet' or config.model.class_name == 'models.fpn_deep.FpnDeep':
        network = factory.create(config.model.class_name)(config, datagen)
    else:
        network = factory.create(config.model.class_name)(config)

    # Load weight file in model
    load_weights(network.model, config)

    if visualization:
        qualitative(network.model, config, datagen)
    else:
        quantitative(network.model, config, datagen)
