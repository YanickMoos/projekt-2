FROM openjdk:21-jdk-slim as builder

WORKDIR /app

COPY .mvn/ .mvn
COPY mvnw pom.xml ./

RUN sed -i 's/\r$//' mvnw
RUN chmod +x mvnw
RUN ./mvnw dependency:go-offline -B

COPY src ./src

RUN ./mvnw package -DskipTests

FROM openjdk:21-slim

WORKDIR /app

COPY --chown=nobody:nogroup model ./model

COPY --from=builder /app/target/playground-0.0.1-SNAPSHOT.jar ./app.jar

EXPOSE 8080

CMD ["java", "-jar", "/app/app.jar"]