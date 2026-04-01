# Lesson: Coaching: Spring AI Part 3 — Retrieval Augmented Generation (RAG)

## Lesson Overview

This is the third and final Spring AI coaching session. We continue building on the `spring-ai-demo` project. In this lesson we cover **RAG — Retrieval Augmented Generation** — the technique that allows you to feed the AI your own private data and have it answer questions based on that data. This is how real enterprise AI applications work: ChatGPT over your own documents, your company FAQ, your product catalogue, or your customer database.

**Prerequisites:** Spring AI basics (Lessons 3.12 and 3.15) — `ChatClient`, system prompts, structured output, conversation memory

## Lesson Objectives

By the end of this lesson, students will be able to:

1. **Explain** what RAG is and why it is needed for real-world AI applications
2. **Load** custom data into a vector store using Spring AI
3. **Build** a RAG-powered endpoint that answers questions based on your own documents

---

## Part 1: The Problem RAG Solves

### What LLMs Don't Know

The LLM models we have been using (GPT-4o-mini) were trained on publicly available data up to a certain date. This means they have no knowledge of:

- **Your company's internal documents** — policies, FAQs, product specs
- **Your application's data** — customer records, orders, inventory
- **Recent events** — anything that happened after their training cutoff
- **Private or sensitive information** — anything that was never public

If you ask GPT-4o-mini "What is our company's refund policy?" it will either make something up or say it doesn't know. Neither is acceptable in a real application.

### The RAG Solution

RAG solves this by giving the AI relevant context from your own data before it answers. The flow works like this:

1. **You load your documents** into a special database called a **Vector Store**
2. **A user asks a question** via your API
3. **Spring AI searches** the vector store for documents that are relevant to the question
4. **Those documents are injected** into the prompt as context before sending it to the LLM
5. **The LLM answers** based on your documents, not just its training data

The result: the AI answers questions accurately using your private data, without you having to share that data permanently with OpenAI.

### What is a Vector Store?

A vector store is a special type of database that stores data as **embeddings** — mathematical representations (arrays of numbers) that capture the meaning of text. When you search a vector store, it finds documents that are semantically similar to your query — meaning it understands meaning, not just keywords.

For example, searching for "How do I cancel my subscription?" would find documents containing "account cancellation", "unsubscribe", or "stop billing" — even if none of them use the exact words from the query.

For this lesson we will use **SimpleVectorStore** — an in-memory vector store provided by Spring AI that requires no external database setup. It is perfect for learning and small applications.

---

## Part 2: Project Setup

### Add the Required Dependency

Open your `spring-ai-demo` project. We need to add one new dependency — the Spring AI vector store advisors library which contains the `QuestionAnswerAdvisor` we will use for RAG.

Add this to your `<dependencies>` block in `pom.xml`:

```xml
<dependency>
  <groupId>org.springframework.ai</groupId>
  <artifactId>spring-ai-advisors-vector-store</artifactId>
</dependency>
```

No version number needed — the BOM we added in Lesson 3.12 manages the version automatically.

Save the file and let Maven reload the dependencies.

### Create Your Knowledge Base

We need some data to feed the AI. Let's create a simple company FAQ file.

Create a new file at `src/main/resources/faq.txt` and add the following content:

```
ACME CRM - Frequently Asked Questions

Q: What is ACME CRM?
A: ACME CRM is a customer relationship management platform designed for small and medium businesses. It helps teams manage customer data, track sales pipelines, and automate follow-ups.

Q: How do I add a new customer?
A: Go to the Customers section, click the "Add Customer" button, fill in the required fields (first name, last name, email), and click Save. The system will automatically generate a unique customer ID.

Q: What is the refund policy?
A: ACME CRM offers a 30-day money-back guarantee. If you are not satisfied within the first 30 days, contact support@acmecrm.com for a full refund. No questions asked.

Q: How do I export customer data?
A: Go to Settings > Data Management > Export. You can export all customer data as a CSV or JSON file. Exports are processed within 24 hours and sent to your registered email.

Q: What are the subscription plans?
A: ACME CRM offers three plans: Starter ($19/month, up to 500 customers), Professional ($49/month, up to 5000 customers), and Enterprise ($149/month, unlimited customers with dedicated support).

Q: How do I reset my password?
A: Click "Forgot Password" on the login page, enter your email address, and you will receive a reset link within 5 minutes. The link expires after 1 hour.

Q: Is my data secure?
A: Yes. All data is encrypted at rest using AES-256 and in transit using TLS 1.3. We are SOC 2 Type II certified and conduct annual security audits.

Q: How do I contact support?
A: Email us at support@acmecrm.com or use the live chat in the app. Our support hours are Monday to Friday, 9am to 6pm SGT. Enterprise customers have access to 24/7 priority support.
```

This is our "knowledge base" — the private data we want the AI to be able to answer questions about.

---

## Part 3: Loading Data into the Vector Store

### Configure the Vector Store Bean

Create a new configuration class `RagConfig.java`.

```java
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.reader.TextReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.SimpleVectorStore;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.Resource;

import java.util.List;

@Configuration
public class RagConfig {

  @Value("classpath:faq.txt")
  private Resource faqResource;

  @Bean
  public VectorStore vectorStore(EmbeddingModel embeddingModel) {
    // Create an in-memory vector store
    SimpleVectorStore store = SimpleVectorStore.builder(embeddingModel).build();

    // Read the FAQ file
    TextReader textReader = new TextReader(faqResource);
    List<Document> documents = textReader.get();

    // Split into smaller chunks for better retrieval
    List<Document> chunks = new TokenTextSplitter().apply(documents);

    // Load chunks into the vector store
    // This is where embeddings are generated — each chunk is converted to a vector
    store.add(chunks);

    System.out.println("✅ FAQ data loaded into vector store — " + chunks.size() + " chunks");
    return store;
  }
}
```

Let's understand what is happening here:

- `EmbeddingModel` — Spring AI auto-configures this bean from the OpenAI starter. It is the model that converts text into vector embeddings (GPT uses `text-embedding-3-small` by default).
- `TextReader` — reads the contents of our `faq.txt` file as a list of `Document` objects.
- `TokenTextSplitter` — splits large documents into smaller chunks. Smaller chunks give better search results because the vector store can pinpoint the exact relevant section rather than returning a huge document.
- `store.add(chunks)` — converts each chunk to a vector embedding and stores it. This happens once at startup.

Run the application. You should see the log message `✅ FAQ data loaded into vector store` in the console.

---

## Part 4: Building the RAG Endpoint

### Create the RAG Controller

Create a new file `RagController.java`.

```java
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.QuestionAnswerAdvisor;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class RagController {

  private final ChatClient chatClient;

  public RagController(ChatClient.Builder chatClientBuilder, VectorStore vectorStore) {
    this.chatClient = chatClientBuilder
        .defaultSystem("You are a helpful customer support assistant for ACME CRM. " +
                       "Answer questions based only on the provided context. " +
                       "If the answer is not in the context, say you don't have that information " +
                       "and suggest contacting support@acmecrm.com.")
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
}
```

Let's break this down:

- `VectorStore vectorStore` — Spring injects the `VectorStore` bean we configured in `RagConfig.java`.
- `QuestionAnswerAdvisor.builder(vectorStore).build()` — this is the RAG advisor. Before every call to the LLM, it automatically searches the vector store for documents relevant to the user's question and injects them into the prompt as context.
- `.defaultSystem("...")` — gives the AI a clear instruction to only answer from the provided context. This prevents the AI from making up answers.
- The endpoint itself is simple — just pass the question and the advisor does all the RAG work automatically.

### Testing the RAG Endpoint

Run the application and test with questions from the FAQ:

```
localhost:8080/faq?question=What is the refund policy?
localhost:8080/faq?question=How much does the Professional plan cost?
localhost:8080/faq?question=How do I add a customer?
```

The AI should answer accurately based on the FAQ content — not from its general training knowledge.

Now try a question that is NOT in the FAQ:

```
localhost:8080/faq?question=Do you support integration with Salesforce?
```

Because we instructed the AI to only answer from context, it should say it doesn't have that information and direct you to support — rather than making something up.

### The Difference Without RAG

To really appreciate what RAG does, compare this with a plain `/chat` call using the same question:

```
localhost:8080/chat?question=What is ACME CRM's refund policy?
```

The plain `/chat` endpoint has no knowledge of ACME CRM — it will either say it doesn't know, or worse, invent a plausible-sounding but completely fabricated answer. This is called **hallucination** and is one of the biggest risks in AI applications.

RAG eliminates hallucination for your domain-specific data by grounding every answer in real, verified documents.

---

### 🧑‍💻 Activity **(20 minutes)**

Add a second knowledge base to your application about a fictional product catalogue.

1. Create `src/main/resources/products.txt` with at least 5 fictional products, each with a name, description, price, and key features. Make it realistic — think electronics, software, or any product category you like.

2. Update `RagConfig.java` to also load the products file into the same vector store.

3. Create a new endpoint `/product-info` in `RagController.java` with a system prompt suitable for a product advisor assistant.

4. Test with questions like:
   - "What products do you have under $100?"
   - "Tell me about [product name]"
   - "What are the key features of [product name]?"

**Hint:** To load a second file, inject it as a second `@Value` resource in `RagConfig.java` and add its chunks to the same `store.add()` call.

---

## Summary

In this session you built a complete RAG pipeline from scratch using Spring AI:

- **SimpleVectorStore** — in-memory vector store, no external database needed for development
- **EmbeddingModel** — auto-configured by Spring AI; converts text to vectors
- **TextReader + TokenTextSplitter** — loads and chunks your documents for better retrieval
- **QuestionAnswerAdvisor** — the RAG advisor that automatically finds relevant context and injects it into every prompt
- Combine with a **system prompt** to constrain the AI to only answer from your documents

This is the foundation of real enterprise AI applications. Production-ready applications swap `SimpleVectorStore` for a proper vector database (PGVector, Pinecone, ChromaDB, etc.) and load thousands of documents — but the code pattern is exactly the same.

### The Spring AI Series — What You've Built

Across the three Spring AI coaching sessions you have gone from zero to building a fully featured AI-powered Spring Boot application:

| Session | What You Built |
|---|---|
| 3.12 | LLM integration, basic chat endpoint, system prompts |
| 3.15 | Structured output (Java objects from AI), conversation memory |
| 3.18 | RAG — AI that answers from your own private documents |

These are the exact same building blocks used in production AI applications at companies today. Well done! 🎉

---

END