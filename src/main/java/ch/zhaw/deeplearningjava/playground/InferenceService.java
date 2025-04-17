package ch.zhaw.deeplearningjava.playground;

import ai.djl.MalformedModelException;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.Model;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.basicmodelzoo.cv.classification.ResNetV1;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.Utils;

import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

@Service
public class InferenceService {

    private static final Logger logger = LoggerFactory.getLogger(InferenceService.class);

    private static final Path MODEL_DIR = Paths.get("model");
    private static final String MODEL_NAME = "fruit-resnet18";

    private static final int IMAGE_WIDTH = 224;
    private static final int IMAGE_HEIGHT = 224;

    private Model model;
    private Predictor<Image, Classifications> predictor;

    @PostConstruct
    public void loadModelAndPredictor() {
        logger.info("Loading model structure and parameters...");
        try {
            Path synsetPath = MODEL_DIR.resolve("synset.txt");
            if (!Files.exists(synsetPath)) {
                logger.error("Synset file not found at: {}", synsetPath);
                throw new IOException("Synset file is missing.");
            }
            final List<String> localSynset = Utils.readLines(synsetPath);
            logger.info("Synset loaded successfully. {} classes.", localSynset.size());

            int numClasses = localSynset.size();
            Block resNet18 = ResNetV1.builder()
                    .setImageShape(new Shape(3, IMAGE_HEIGHT, IMAGE_WIDTH))
                    .setNumLayers(18)
                    .setOutSize(numClasses)
                    .build();

            this.model = Model.newInstance(MODEL_NAME);
            this.model.setBlock(resNet18);

            this.model.load(MODEL_DIR);
            logger.info("Model parameters loaded successfully from directory: {}", MODEL_DIR);

            Translator<Image, Classifications> translator = createManualTranslator(localSynset);

            predictor = model.newPredictor(translator);
            logger.info("Predictor created successfully.");

        } catch (IOException | ModelException e) {
            logger.error("Error loading model or creating predictor", e);
            throw new RuntimeException("Failed to initialize InferenceService", e);
        }
    }

    private Translator<Image, Classifications> createManualTranslator(List<String> synset) {
         return new Translator<Image, Classifications>() {
             private final List<String> classes = synset;

             @Override
             public Classifications processOutput(TranslatorContext ctx, NDList list) {
                 NDArray probabilities = list.singletonOrThrow().softmax(0);
                 return new Classifications(this.classes, probabilities);
             }

             @Override
             public NDList processInput(TranslatorContext ctx, Image input) {
                 NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.COLOR);
                 array = NDImageUtils.resize(array, IMAGE_WIDTH, IMAGE_HEIGHT);
                 if (array.getShape().dimension() == 3) {
                    array = array.transpose(2, 0, 1);
                 }
                 array = array.toType(DataType.FLOAT32, false).div(255.0f);
                 return new NDList(array);
             }
        };
    }

    public Classifications predict(MultipartFile imageFile) throws IOException, TranslateException {
        if (predictor == null) {
            logger.error("Predictor is not initialized!");
            throw new IllegalStateException("Inference Service is not ready.");
        }
        if (imageFile.isEmpty()) {
            throw new IllegalArgumentException("Image file is empty.");
        }

        try (InputStream is = imageFile.getInputStream()) {
            Image image = ImageFactory.getInstance().fromInputStream(is);
            logger.info("Received image for prediction: {}", imageFile.getOriginalFilename());
            Classifications classifications = predictor.predict(image);
            logger.info("Prediction result: {}", classifications);
            return classifications;
        }
    }

    @PreDestroy
    public void cleanUp() {
        logger.info("Closing model and predictor...");
        if (predictor != null) {
        }
        if (model != null) {
            model.close();
        }
        logger.info("InferenceService cleaned up.");
    }
}
