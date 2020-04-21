from evaluater.quantitative import evaluate_accuracy, evaluate_speed
from evaluater.qualitative import visualize_results
from utils import factory
from data_generators.segmentation_data_generator import SegmentationDataGenerator


def load_weights(model, config):
    """Load weights file if required by configuration
    """
    if not hasattr(config, 'evaluation') or not hasattr(config.evaluation, 'weights_file'):
        return

    print('Load weight file : ', config.evaluation.weights_file)
    model.load_weights(config.evaluation.weights_file)


def quantitative(model, config, datagen):
    """Quantitative evaluation 
    """
    print('################# PROCESSING METRICS FOR ACCURACY ###############')
    Aiou, class_Aiou, pixel_acc, mean_pixel_acc = evaluate_accuracy(
        model, config, datagen)

    print('################# PROCESSING METRICS FOR INFERENCE TIME ###############')
    total_time, fps = evaluate_speed(model, config, datagen)

    print('################# Results : ##################')
    print('Pixel accuracy : ', pixel_acc)
    print('Mean pixel accuracy : ', mean_pixel_acc)
    print('Mean IoU : ', Aiou)
    print('Per-class mIoUs : ', class_Aiou)


def qualitative(model, config, datagen):
    """Qualitative evaluation (visualize all the results)
    """
    print('################ BEGINNING VISUALIZATION ############')
    print('--- Press A to stop visualization')
    print('--- Press any other key to continue')
    visualize_results(model, config, datagen)


def evaluation(config, visualization=False):
    # dynamic model instanciation
    model = factory.create(config.model.class_name)(config)

    # Data generator creation
    datagen = SegmentationDataGenerator(config, is_training_set=False)

    # Load weight file in model
    load_weights(model, config)

    if visualization:
        qualitative(model, config, datagen)
    else:
        quantitative(model, config, datagen)
