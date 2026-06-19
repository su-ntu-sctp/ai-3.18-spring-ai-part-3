# Lesson 3.18 — Activity Solutions

---

## Activity — Add a Product Catalogue Knowledge Base

### Step 1: Create `products.txt`

Create a new file at `src/main/resources/products.txt`:

```
ACME Tech Store - Product Catalogue

Product: UltraBook Pro 15
Description: A lightweight, high-performance laptop designed for professionals.
Price: $1,299
Key Features: Intel Core i7, 16GB RAM, 512GB SSD, 15.6" 4K display, 12-hour battery life, weighs only 1.4kg.

Product: SmartHub 360
Description: A smart home hub that connects and controls all your smart devices from one place.
Price: $89
Key Features: Compatible with Alexa, Google Home, and Apple HomeKit. Supports up to 50 devices. Built-in Wi-Fi and Bluetooth.

Product: ProSound Wireless Headphones
Description: Premium noise-cancelling over-ear headphones for audiophiles and remote workers.
Price: $249
Key Features: Active noise cancellation, 30-hour battery, hi-res audio certified, foldable design, USB-C charging.

Product: SnapCam 4K
Description: A compact action camera built for adventure and content creation.
Price: $179
Key Features: 4K/60fps video, waterproof up to 10m, built-in image stabilisation, 2-inch touchscreen, voice control.

Product: DeskPad Ultra
Description: A productivity-focused smart desk mat with wireless charging and USB hub.
Price: $69
Key Features: 15W wireless charging pad, 3x USB-A ports, 1x USB-C port, non-slip base, 80x40cm surface, cable management slot.

Product: CloudSync Storage 2TB
Description: A personal cloud storage device for home and small office use.
Price: $149
Key Features: 2TB capacity, automatic backup, remote access via mobile app, supports up to 5 users, RAID-ready.
```

---

### Step 2: Update `RagConfig.java`

Load both `faq.txt` and `products.txt` into the same vector store:

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
    List<Document> faqChunks = new TokenTextSplitter()
        .apply(new TextReader("classpath:faq.txt").get());

    // Read and chunk the products file
    List<Document> productChunks = new TokenTextSplitter()
        .apply(new TextReader("classpath:products.txt").get());

    // Combine all chunks into one list
    List<Document> allChunks = new ArrayList<>();
    allChunks.addAll(faqChunks);
    allChunks.addAll(productChunks);

    // Load all chunks into the vector store
    store.add(allChunks);

    System.out.println("✅ Data loaded into vector store — " + allChunks.size() + " chunks total");
    return store;
  }
}
```

---

### Step 3: Add `/product-info` endpoint to `RagController.java`

Add a second `ChatClient` for the product advisor, or add a new endpoint using the same `chatClient`. The cleanest approach for this activity is to add a second endpoint with its own inline system prompt:

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

  private final ChatClient faqChatClient;
  private final ChatClient productChatClient;

  public RagController(ChatClient.Builder chatClientBuilder, VectorStore vectorStore) {

    // FAQ assistant
    this.faqChatClient = chatClientBuilder
        .defaultSystem("You are a helpful customer support assistant for ACME CRM. " +
                       "Answer questions based only on the provided context. " +
                       "If the answer is not in the context, say you don't have that information " +
                       "and suggest contacting support@acmecrm.com.")
        .defaultAdvisors(QuestionAnswerAdvisor.builder(vectorStore).build())
        .build();

    // Product advisor assistant
    this.productChatClient = chatClientBuilder
        .defaultSystem("You are a helpful product advisor for ACME Tech Store. " +
                       "Answer questions about products based only on the provided context. " +
                       "Include relevant details like price and key features in your answers. " +
                       "If the product is not in the catalogue, say you don't have that information.")
        .defaultAdvisors(QuestionAnswerAdvisor.builder(vectorStore).build())
        .build();
  }

  @GetMapping("/faq")
  public String faq(@RequestParam String question) {
    return faqChatClient.prompt()
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

---

### Step 4: Test the `/product-info` endpoint

```
localhost:8080/product-info?question=What products do you have under $100?
localhost:8080/product-info?question=Tell me about the UltraBook Pro 15
localhost:8080/product-info?question=What are the key features of the ProSound headphones?
localhost:8080/product-info?question=Do you sell gaming keyboards?
```

The last question is not in the catalogue — the AI should say it doesn't have that information rather than making something up.

---

### Expected Behaviour

- Questions about FAQ topics (`/faq`) → answered from `faq.txt`
- Questions about products (`/product-info`) → answered from `products.txt`
- Both endpoints draw from the **same vector store** — Spring AI's similarity search retrieves the most relevant chunks regardless of which file they came from
- Out-of-scope questions → AI responds that it doesn't have that information

---

END