# agenticRAGMentor
HackAI on devpost
# Agentic RAG for Enhanced Learning in Jetson Robotics and Deep Learning

## Introduction
As an NVIDIA Ambassador, I aim to leverage my extensive knowledge and experience with Jetson platforms to mentor and guide students in mastering robotics and deep learning. By implementing Agentic RAG (Retrieval Augmented Generation), I can provide a personalized and dynamic learning experience that empowers students to achieve their full potential.

## Project Overview
Agentic RAG combines retrieval-based and generation-based approaches to create an AI that can assist students in real-time, providing accurate and relevant information based on their queries. This project will utilize NVIDIA's advanced AI tools and Jetson platforms to develop a robust mentorship system.

## Objectives
1. Enhance student learning through personalized AI assistance.
2. Improve accessibility to high-quality educational resources.
3. Foster a deeper understanding of robotics and deep learning concepts.

## Methodology
- **Data Collection**: Gather educational materials and resources related to Jetson Robotics and Deep Learning.
- **AI Development**: Implement Agentic RAG using NVIDIA AI Workbench to create an intelligent assistant capable of understanding and responding to student queries.
- **Integration**: Incorporate the AI assistant into the Jetson Robotics and Deep Learning curriculum, providing real-time support and guidance to students.

## Courses and Topics Covered
- **Introduction to Jetson Nano**
- **Jetson Xavier NX: Advanced Robotics**
- **Deep Learning Fundamentals**
- **Computer Vision with Jetson**
- **Reinforcement Learning with Jetson**
- **Autonomous Systems using Jetson**
- **Edge AI and Embedded Systems**

## Expected Outcomes
- Increased student engagement and participation in Jetson Robotics and Deep Learning courses.
- Enhanced understanding and application of robotics and AI concepts.
- Improved problem-solving skills through real-time AI assistance.

- # Agentic Retrieval-Augmented Generation (RAG) Pipeline in NVIDIA Workbench

This project implements an Agentic Retrieval-Augmented Generation (RAG) pipeline within NVIDIA Workbench, specifically tailored to handle knowledge documents from NVIDIA Jetson courses. The system ingests, indexes, retrieves, and generates responses based on NVIDIA Jetson knowledge using a fine-tuned GPT model.

## Architecture Overview

[Agentic RAG Pipeline](./Architecture Diagram RAG.png) <!-- Add a reference to the PlantUML generated architecture diagram image if available -->

The architecture consists of the following components:

### 1. **User**
   - The end-user interacts with the system by submitting queries related to the NVIDIA Jetson knowledgebase.

### 2. **NVIDIA Workbench**
   - The environment where all the pipeline components operate.

   - **Agentic RAG Orchestrator**: 
     - Coordinates the pipeline flow. It accepts queries from the user, initiates document retrieval, and generates responses.
   
   - **Document Ingestion Pipeline**: 
     - Processes and ingests knowledge documents from NVIDIA Jetson courses and stores them in the knowledgebase.

   - **Document Indexing**:
     - Indexes ingested documents for efficient retrieval and stores them in the knowledge database.

   - **Document Retriever**: 
     - Searches the index for relevant documents based on the user's query and retrieves them from the knowledge database.
   
   - **NVIDIA Jetson Knowledgebase**: 
     - Stores the course-related knowledge documents from NVIDIA Jetson.

   - **NVIDIA GPT Model**: 
     - A large language model fine-tuned with NVIDIA Jetson knowledge that generates contextual responses by combining the user query with retrieved documents.

   - **Response Generator**: 
     - Generates a response for the user by combining relevant documents and the GPT model's output.

   - **Knowledge Database**: 
     - Stores the indexed documents for fast and efficient search and retrieval operations.

## Workflow

1. **User Query**: The user sends a query related to the NVIDIA Jetson courses.
   
2. **Ingestion**: The `Document Ingestion Pipeline` processes and ingests new knowledge documents from NVIDIA Jetson courses into the `NVIDIA Jetson Knowledgebase`.

3. **Indexing**: The `Document Indexing` component indexes these ingested documents and stores the indices in the `Knowledge Database`.

4. **Retrieval**: The `Agentic RAG Orchestrator` sends the user's query to the `Document Retriever`, which searches the `Knowledge Database` for relevant documents.

5. **Response Generation**: The retrieved documents are passed to the `NVIDIA GPT Model` and combined with the query to generate a detailed response through the `Response Generator`.

6. **Result**: The generated response is returned to the user by the orchestrator.

## Installation

To run this project in the NVIDIA Workbench environment, follow these steps:

1. **Prerequisites**:
   - NVIDIA Workbench installed.
   - Access to the NVIDIA Jetson course dataset.
   - A fine-tuned GPT model with NVIDIA Jetson knowledge.

2. **Setting up the Environment**:
   - Clone the repository:
     ```bash
     git clone https://github.com/your-repo/agentic-rag-pipeline.git
     ```
   - Install necessary dependencies:
     ```bash
     cd agentic-rag-pipeline
     pip install -r requirements.txt
     ```

3. **Configuration**:
   - Set the environment variables for your NVIDIA Workbench setup in `.env`.
   
4. **Running the Pipeline**:
   - Start the ingestion process:
     ```bash
     python ingest.py
     ```
   - Index the documents:
     ```bash
     python index.py
     ```
   - Start the RAG pipeline orchestrator:
     ```bash
     python orchestrator.py
     ```

## Usage

Once the pipeline is running, users can submit queries related to NVIDIA Jetson courses through a provided API endpoint, and the system will return generated responses based on the ingested knowledge.

## Contribution

If you'd like to contribute to the project, please follow the standard process for pull requests:

1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.



## Resources
- **NVIDIA Teaching Kit for Edge AI and Robotics Educators**: This kit includes lecture slides, hands-on labs, and online courses that provide the foundation for understanding and building hands-on expertise in AI and GPU computing. It's a valuable resource for educators looking to integrate AI and robotics into their curriculum.

## Conclusion
By integrating Agentic RAG into the Jetson Robotics and Deep Learning curriculum, I aim to create a transformative learning experience for students. This project will not only enhance their technical skills but also inspire them to explore the limitless possibilities of AI and robotics.

