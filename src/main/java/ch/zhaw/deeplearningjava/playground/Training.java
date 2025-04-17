package ch.zhaw.deeplearningjava.playground;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.ImageFolder;
import ai.djl.metric.Metrics;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import ai.djl.basicmodelzoo.cv.classification.ResNetV1;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

public class Training {

    private static final int BATCH_SIZE = 32;
    private static final int EPOCHS = 5;
    private static final int IMAGE_WIDTH = 224;
    private static final int IMAGE_HEIGHT = 224;
    private static final String MODEL_NAME = "fruit-resnet18";
    private static final Path MODEL_DIR = Paths.get("model");
    private static final Path DATASET_DIR = Paths.get("dataset/fruits-custom");

    private static final Logger logger = LoggerFactory.getLogger(Training.class);

    public static void main(String[] args) throws IOException, TranslateException {
        logger.info("Starting dataset preparation...");

        ImageFolder dataset = initDataset(DATASET_DIR);

        logger.info("Splitting dataset...");
        Dataset[] splits = dataset.randomSplit(8, 2);
        Dataset trainingSet = splits[0];
        Dataset validationSet = splits[1];

        int numClasses = dataset.getSynset().size();
        logger.info("Dataset prepared.");
        logger.info("Number of classes: {}", numClasses);

        Block resNet18 = ResNetV1.builder()
                .setImageShape(new Shape(3, IMAGE_HEIGHT, IMAGE_WIDTH))
                .setNumLayers(18)
                .setOutSize(numClasses)
                .build();

        try (Model model = Model.newInstance(MODEL_NAME)) {
            model.setBlock(resNet18);
            model.setProperty("numClasses", String.valueOf(numClasses));

            DefaultTrainingConfig config = setupTrainingConfig();

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.setMetrics(new Metrics());

                Shape inputShape = new Shape(BATCH_SIZE, 3, IMAGE_HEIGHT, IMAGE_WIDTH);
                trainer.initialize(inputShape);
                logger.info("Trainer initialized. Input shape: {}", inputShape);

                logger.info("Starting training for {} epochs...", EPOCHS);
                EasyTrain.fit(trainer, EPOCHS, trainingSet, validationSet);
                logger.info("Training finished.");

                TrainingResult result = trainer.getTrainingResult();
                saveTrainingResult(model, result);

                model.setProperty("Traced", "true");
                logger.info("Setting Traced=true to attempt saving as TorchScript (.pt)");

                saveModel(model, dataset.getSynset());

            }
        }
    }

    private static ImageFolder initDataset(Path datasetRoot) throws IOException, TranslateException {
        ImageFolder dataset = ImageFolder.builder()
                .setRepositoryPath(datasetRoot)
                .optMaxDepth(10)
                .addTransform(new Resize(IMAGE_WIDTH, IMAGE_HEIGHT))
                .addTransform(new ToTensor())
                .setSampling(BATCH_SIZE, true)
                .build();
        dataset.prepare(new ProgressBar());
        return dataset;
    }

    private static DefaultTrainingConfig setupTrainingConfig() {
        Loss loss = Loss.softmaxCrossEntropyLoss();
        return new DefaultTrainingConfig(loss)
                .addEvaluator(new Accuracy())
                .addTrainingListeners(TrainingListener.Defaults.logging());
    }

    private static void saveTrainingResult(Model model, TrainingResult result) {
        model.setProperty("Epoch", String.valueOf(EPOCHS));
        if (result.getValidateEvaluation("Accuracy") != null) {
            model.setProperty("Accuracy", String.format("%.5f", result.getValidateEvaluation("Accuracy")));
        }
        if (result.getValidateLoss() != null) {
            model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));
        }
        logger.info("Final Validation Accuracy: {}", model.getProperty("Accuracy", "N/A"));
        logger.info("Final Validation Loss: {}", model.getProperty("Loss", "N/A"));
    }

    private static void saveModel(Model model, List<String> synset) throws IOException {
        logger.info("Saving model (with Traced=true) and synset to: {}", MODEL_DIR.toAbsolutePath());
        Files.createDirectories(MODEL_DIR);
        model.save(MODEL_DIR, MODEL_NAME);
        Path synsetFile = MODEL_DIR.resolve("synset.txt");
        Files.write(synsetFile, synset);
        logger.info("Model save attempted (check for .pt file) and synset saved successfully.");
    }
}
