# Lesson 3.18: Spring AI Part 3 — Retrieval Augmented Generation (RAG)

## Lesson Overview

This is the third and final Spring AI lesson. We continue building on the `spring-ai-demo` project. In this lesson we cover **RAG — Retrieval Augmented Generation** — the technique that allows you to feed the AI your own private data and have it answer questions based on that data. This is how real enterprise AI applications work: ChatGPT over your own documents, your company FAQ, your product catalogue, or your customer database.

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

### Key Concepts

Before we write any code, it is important to understand the core concepts that make RAG work.

#### Vector Store / Vector Database

A vector store is a special type of database designed for AI applications. Unlike a regular database that stores rows and columns and searches by exact match, a vector store stores data as **embeddings** — arrays of numbers that represent the *meaning* of text. When you search a vector store, it finds documents that are *semantically similar* to your query.

For example, searching for "How do I cancel my subscription?" would find documents containing "account cancellation", "unsubscribe", or "stop billing" — even if none of them use the exact words from the query. A regular SQL `WHERE` clause could never do this.

For this lesson we use **SimpleVectorStore** — an in-memory vector store provided by Spring AI that requires no external database setup. It is perfect for learning. In production, teams use dedicated vector databases such as PGVector (PostgreSQL extension), Pinecone, ChromaDB, or Qdrant.

#### Embeddings

An embedding is a mathematical representation of text as an array of numbers (a vector). The key property is that text with similar *meaning* produces vectors that are numerically close to each other in vector space.

For example:
- "Cancel my account" and "How do I unsubscribe?" are semantically similar → their vectors will be close together
- "Cancel my account" and "What is the weather today?" are unrelated → their vectors will be far apart

Spring AI uses OpenAI's `text-embedding-3-small` model by default to generate embeddings. This is a separate model from the chat model (GPT-4o-mini) — its only job is to convert text into vectors.

> **Instructor insight:** The `EmbeddingModel` bean is auto-configured by Spring AI from the OpenAI starter. You do not need to configure it manually. When you call `store.add(chunks)`, Spring AI calls the embedding model internally to convert each chunk to a vector before storing it.

#### Chunking

When we load a document into the vector store, we first split it into smaller pieces called **chunks**. This is done by `TokenTextSplitter`.

Why chunk? Because if we store the entire document as one vector, the similarity search returns the whole document even when only one paragraph is relevant. Smaller chunks allow the vector store to pinpoint exactly which section answers the question, resulting in more accurate and focused responses.

#### Similarity Search

When a user asks a question, Spring AI converts that question into an embedding (vector) and searches the vector store for chunks whose vectors are closest to the question vector. This is called **cosine similarity search**. The top matching chunks are retrieved and injected into the prompt as context for the LLM to answer from.

---

## Part 2: Project Setup

### Add the Required Dependency

Open your `spring-ai-demo` project. We need to add one new dependency — the Spring AI vector store advisor library which contains the `QuestionAnswerAdvisor` we will use for RAG.

Add this to your `<dependencies>` block in `pom.xml`:

```xml
<dependency>
  <groupId>org.springframework.ai</groupId>
  <artifactId>spring-ai-vector-store-advisor</artifactId>
</dependency>
```

> **Note:** The artifact ID is `spring-ai-vector-store-advisor` — not `spring-ai-advisors-vector-store`. The module was renamed in Spring AI 2.0 to align with naming conventions. Using the old name will cause a build failure.

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
package sg.edu.ntu.spring_ai_demo;

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

    // Read the FAQ file directly using the classpath path string
    TextReader textReader = new TextReader("classpath:faq.txt");
    List<Document> documents = textReader.get();

    // Split into smaller chunks for better retrieval
    List<Document> chunks = TokenTextSplitter.builder().build().apply(documents);

    // Load chunks into the vector store
    // This is where embeddings are generated — each chunk is converted to a vector
    store.add(chunks);

    System.out.println("✅ FAQ data loaded into vector store — " + chunks.size() + " chunks");
    return store;
  }
}
```

> **Note:** `TextReader` accepts either a plain classpath string or a Spring `Resource` — we're using the string form here for simplicity.

Let's understand what is happening here:

- **`EmbeddingModel`** — Spring AI auto-configures this bean from the OpenAI starter. It is the model that converts text into vector embeddings. OpenAI uses `text-embedding-3-small` by default. You do not need to configure it manually — Spring AI injects it automatically.
- **`TextReader`** — reads the contents of `faq.txt` as a list of `Document` objects.
- **`TokenTextSplitter`** — splits the document into smaller chunks. Smaller chunks give better search results because the vector store can pinpoint exactly the relevant section rather than returning a huge document.
- **`store.add(chunks)`** — converts each chunk to a vector embedding by calling the `EmbeddingModel` and stores both the vector and the original text. This happens once at startup.

Run the application. You should see the log message `✅ FAQ data loaded into vector store` in the console.

---

## Part 4: Building the RAG Endpoint

### Create the RAG Controller

Create a new file `RagController.java`.

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

> **Note:** `@RestController` is required — without it, Spring will not register this class as a controller and all requests to `/faq` will return a `404 Not Found`.

Let's break this down:

- **`VectorStore vectorStore`** — Spring injects the `VectorStore` bean we configured in `RagConfig.java`.
- **`QuestionAnswerAdvisor.builder(vectorStore).build()`** — this is the RAG advisor. Before every call to the LLM, it automatically:
  1. Converts the user's question into an embedding
  2. Searches the vector store for the most relevant chunks
  3. Injects those chunks into the prompt as context
- **`.defaultSystem("...")`** — instructs the AI to only answer from the provided context. This is important — without this instruction, the AI may ignore the context and answer from its general training knowledge.
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

2. Update `RagConfig.java` to also load the products file into the same vector store. Add a second `TextReader` for `"classpath:products.txt"` and include its chunks in the same `store.add()` call.

3. Create a new endpoint `/product-info` in `RagController.java` with a system prompt suitable for a product advisor assistant.

4. Test with questions like:
   - "What products do you have under $100?"
   - "Tell me about [product name]"
   - "What are the key features of [product name]?"

**Hint:** To load both files, read and chunk each one separately, then combine the lists before adding to the store:

```java
List<Document> faqChunks = TokenTextSplitter.builder().build().apply(new TextReader("classpath:faq.txt").get());
List<Document> productChunks = TokenTextSplitter.builder().build().apply(new TextReader("classpath:products.txt").get());

List<Document> allChunks = new ArrayList<>();
allChunks.addAll(faqChunks);
allChunks.addAll(productChunks);

store.add(allChunks);
```

---

## Summary

In this session you built a complete RAG pipeline from scratch using Spring AI:

- **SimpleVectorStore** — in-memory vector store, no external database needed for development
- **EmbeddingModel** — auto-configured by Spring AI; converts text to vectors using `text-embedding-3-small`
- **TextReader + TokenTextSplitter** — loads and chunks your documents for better retrieval
- **QuestionAnswerAdvisor** — the RAG advisor that automatically finds relevant context and injects it into every prompt
- Combine with a **system prompt** to constrain the AI to only answer from your documents

This is the foundation of real enterprise AI applications. Production-ready applications swap `SimpleVectorStore` for a proper vector database (PGVector, Pinecone, ChromaDB, etc.) and load thousands of documents — but the code pattern is exactly the same.

### The Spring AI Series — What You've Built

Across the three Spring AI lessons you have gone from zero to building a fully featured AI-powered Spring Boot application:

| Session | What You Built |
|---|---|
| 3.12 | LLM integration, basic chat endpoint, system prompts |
| 3.15 | Structured output (Java objects from AI), conversation memory |
| 3.18 | RAG — AI that answers from your own private documents |

These are the exact same building blocks used in production AI applications at companies today. Well done! 🎉

---

END