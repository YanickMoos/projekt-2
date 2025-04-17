package ch.zhaw.deeplearningjava.playground;

import ai.djl.modality.Classifications;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.ApplicationRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.core.io.ClassPathResource;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.util.FileCopyUtils;

import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

@SpringBootApplication
public class PlaygroundApplication {

    private static final Logger logger = LoggerFactory.getLogger(PlaygroundApplication.class);

    public static void main(String[] args) {
        SpringApplication.run(PlaygroundApplication.class, args);
    }

    @Bean
    ApplicationRunner init(InferenceService inferenceService) {
        return args -> {
            logger.info("---------- STARTING INTERNAL INFERENCE TEST ----------");
            try {
                String testImageName = "apple.jpg";

                logger.info("Attempting to load test image: {}", testImageName);

                ClassPathResource resource = new ClassPathResource("test-images/" + testImageName);
                if (!resource.exists()) {
                    logger.error("Test image not found in classpath: test-images/{}", testImageName);
                    return;
                }

                MockMultipartFile mockFile = new MockMultipartFile(
                        "image",
                        testImageName,
                        Files.probeContentType(Path.of(resource.getURI())),
                        resource.getInputStream()
                );

                logger.info("Created MockMultipartFile for {}", mockFile.getOriginalFilename());

                logger.info("Calling InferenceService.predict()...");
                Classifications result = inferenceService.predict(mockFile);

                logger.info("---------- INTERNAL INFERENCE TEST RESULT ----------");
                logger.info("Prediction for {}: {}", testImageName, result);
                if (result != null && !result.items().isEmpty()) {
                    logger.info("Top prediction: {}", result.best());
                }
                logger.info("----------------------------------------------------");

            } catch (Exception e) {
                logger.error("---------- INTERNAL INFERENCE TEST FAILED ----------", e);
            }
        };
    }
}
