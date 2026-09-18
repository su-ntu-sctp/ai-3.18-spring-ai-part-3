# Lesson 3.18 — Activity Solution

**Activity:** Add a product catalogue as a second knowledge base, and build a `/product-info` endpoint that answers from it.

---

## Step 1 — `src/main/resources/products.txt`

Provided in lesson.md — students copy it in as-is.

---

## Step 2 — Updated `RagConfig.java`

Both files are read and chunked using the same three steps, then combined into one list before a single `store.add()` call.

```java
package sg.edu.ntu.spring_ai_demo;

import java.util.ArrayList;
import java.util.List;

import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.reader.TextReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.SimpleVectorStore;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class RagConfig {

  @Bean
  public VectorStore vectorStore(EmbeddingModel embeddingModel) {
    // Create an in-memory vector store
    SimpleVectorStore store = SimpleVectorStore.builder(embeddingModel).build();

    // Read and chunk the FAQ file
    TextReader faqReader = new TextReader("classpath:faq.txt");
    List<Document> faqDocuments = faqReader.get();
    List<Document> faqChunks = TokenTextSplitter.builder().build().apply(faqDocuments);

    // Read and chunk the Products file — same three steps
    TextReader productReader = new TextReader("classpath:products.txt");
    List<Document> productDocuments = productReader.get();
    List<Document> productChunks = TokenTextSplitter.builder().build().apply(productDocuments);

    // Combine both sets of chunks, then load them in one call
    List<Document> allChunks = new ArrayList<>();
    allChunks.addAll(faqChunks);
    allChunks.addAll(productChunks);

    store.add(allChunks);

    System.out.println("✅ FAQ + Products loaded into vector store — " + allChunks.size() + " chunks");
    return store;
  }
}
```

---

## Step 3 — Updated `RagController.java`

A second `ChatClient` is built in the same constructor, using the same `vectorStore` but its own system prompt.

```java
package sg.edu.ntu.spring_ai_demo;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.vectorstore.QuestionAnswerAdvisor;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class RagController {

  private final ChatClient chatClient;
  private final ChatClient productChatClient;

  public RagController(ChatClient.Builder chatClientBuilder, VectorStore vectorStore) {

    // Existing FAQ support assistant
    this.chatClient = chatClientBuilder
        .defaultSystem("You are a helpful customer support assistant for ACME CRM. " +
                       "Answer questions based only on the provided context. " +
                       "If the answer is not in the context, say you don't have that information " +
                       "and suggest contacting support@acmecrm.com.")
        .defaultAdvisors(QuestionAnswerAdvisor.builder(vectorStore).build())
        .build();

    // New product advisor assistant — same vector store, different system prompt
    this.productChatClient = chatClientBuilder
        .defaultSystem("You are a friendly product advisor for ACME Tech. " +
                       "Answer questions about our products using only the provided context. " +
                       "If a product or detail isn't in the context, say you don't have that " +
                       "information rather than guessing.")
        .defaultAdvisors(QuestionAnswerAdvisor.builder(vectorStore).build())
        .build();
  }

  @GetMapping("/faq")
  public String faq(@RequestParam String question) {
    return chatClient.prompt()
        .user(question)
        .call()
        .content();
  }

  @GetMapping("/product-info")
  public String productInfo(@RequestParam String question) {
    return productChatClient.prompt()
        .user(question)
        .call()
        .content();
  }
}
```

> **Note:** Students' system prompts will differ in wording — that's fine. What matters is that it constrains the AI to the provided context and tells it not to guess.

---

## Step 4 — Testing

```
localhost:8080/product-info?question=What products do you have under $100?
localhost:8080/product-info?question=Tell me about the PulseBuds Pro
localhost:8080/product-info?question=What are the key features of the OrbitLock Smart Padlock?
```

---

## Discussion point

Try this on the product endpoint:

```
localhost:8080/product-info?question=What is the refund policy?
```

It may well answer from the FAQ chunks, because both knowledge bases share one vector store. Retrieval is based purely on vector similarity — it has no idea a chunk "belongs to" products or FAQ.

In production this is solved with **metadata filtering**: each document is tagged (e.g. `category=product`), and the similarity search is filtered to only that category. Worth mentioning as a forward pointer, though it's beyond this lesson's scope.
