package ch.zhaw.deeplearningjava.playground;

import ai.djl.modality.Classifications;
import ai.djl.translate.TranslateException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@RestController
public class ClassificationController {

    private static final Logger logger = LoggerFactory.getLogger(ClassificationController.class);

    @Autowired
    private InferenceService inferenceService;

    @GetMapping("/ping")
    public String ping() {
        logger.info("Received ping request");
        return "Classification Controller is running!";
    }

    @PostMapping("/predict")
    public ResponseEntity<String> predict(@RequestParam("image") MultipartFile imageFile) {
        logger.info("Received prediction request for image: {}", imageFile.getOriginalFilename());

        if (imageFile.isEmpty()) {
            logger.warn("Received empty image file.");
            return ResponseEntity.badRequest().body("Image file is empty.");
        }

        try {
            Classifications result = inferenceService.predict(imageFile);
            return ResponseEntity.ok(result.toJson());

        } catch (IOException e) {
            logger.error("Error reading image file.", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("Error processing image file.");
        } catch (TranslateException e) {
            logger.error("Prediction failed.", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("Prediction failed: " + e.getMessage());
        } catch (IllegalStateException e) {
            logger.error("Inference service not ready.", e);
            return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE).body("Service not ready: " + e.getMessage());
        } catch (Exception e) {
            logger.error("An unexpected error occurred during prediction.", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("An unexpected error occurred.");
        }
    }
}
