"""
generate_dataset.py
-------------------
Generates a labeled prompt quality dataset for training.
Creates 1000 samples (500 low-quality + 500 high-quality).
Saves to dataset/prompt_dataset.csv

Run: python generate_dataset.py
"""

import pandas as pd
import random
import os

# ============================================================
# BASE TOPICS
# ============================================================
topics = [
    "artificial intelligence", "machine learning", "deep learning",
    "data science", "Python programming", "blockchain technology",
    "cloud computing", "cybersecurity", "internet of things",
    "big data", "digital marketing", "web development",
    "mobile app development", "software engineering", "database management",
    "computer networking", "operating systems", "algorithms",
    "data structures", "object oriented programming",
    "climate change", "renewable energy", "space exploration",
    "genetic engineering", "quantum computing", "robotics",
    "virtual reality", "augmented reality", "5G technology",
    "edge computing", "natural language processing", "computer vision",
    "neural networks", "reinforcement learning", "transfer learning",
    "API development", "microservices", "DevOps",
    "agile methodology", "project management", "UI UX design",
    "ecommerce", "supply chain management", "financial technology",
    "healthcare technology", "autonomous vehicles", "smart cities",
    "sustainable development", "electric vehicles", "photosynthesis",
    "human computer interaction", "information security",
    "embedded systems", "signal processing", "game development",
    "cryptocurrency", "decentralized finance", "cloud storage",
    "containerization", "serverless computing", "data visualization",
    "business intelligence", "predictive analytics", "sentiment analysis",
    "recommendation systems", "fraud detection", "image recognition",
    "speech recognition", "text summarization", "chatbot development",
    "automation testing", "continuous integration", "version control",
    "Linux administration", "Windows server", "network security",
    "penetration testing", "ethical hacking", "digital forensics",
    "data warehousing", "ETL processes", "data mining",
    "statistical analysis", "probability theory", "linear algebra",
    "calculus applications", "optimization techniques", "graph theory",
    "compiler design", "programming paradigms", "functional programming",
    "concurrent programming", "distributed systems", "parallel computing",
    "software architecture", "design patterns", "clean code principles",
    "test driven development", "behavior driven development",
    "REST API design", "GraphQL", "WebSocket programming",
    "Docker containers", "Kubernetes orchestration", "CI CD pipelines",
    "machine learning operations", "model deployment", "feature engineering",
    "time series analysis", "anomaly detection", "clustering algorithms",
]

# ============================================================
# LOW QUALITY PROMPT TEMPLATES
# ============================================================
low_quality_templates = [
    "Explain {topic}",
    "Tell me about {topic}",
    "What is {topic}",
    "Define {topic}",
    "Describe {topic}",
    "{topic}",
    "Write about {topic}",
    "{topic} explain",
    "Help me with {topic}",
    "I need info on {topic}",
    "Give me {topic} details",
    "Talk about {topic}",
    "Brief on {topic}",
    "{topic} overview",
    "{topic}?",
    "What about {topic}",
    "Something about {topic}",
    "Need to know {topic}",
    "Info {topic}",
    "Plz explain {topic}",
]

# ============================================================
# HIGH QUALITY PROMPT TEMPLATES
# ============================================================
high_quality_templates = [
    "Explain {topic} in simple terms. Provide 3 real-world examples. Use bullet points. Limit your response to 200 words.",
    "You are an expert professor. Explain {topic} to a beginner student step-by-step with practical examples and a summary at the end.",
    "Write a detailed explanation of {topic} covering: 1) Definition, 2) Key concepts, 3) Real-world applications, 4) Advantages and disadvantages. Use clear headings.",
    "Describe {topic} in a structured format. Include an introduction, 3 main points with examples, and a conclusion. Target audience: college students.",
    "Explain {topic} as if teaching a 10-year-old. Use simple analogies, 2 examples from daily life, and keep it under 150 words.",
    "Compare and contrast {topic} with its closest alternative. Present your answer in a table format with at least 5 comparison points.",
    "Write a comprehensive guide on {topic}. Include: prerequisites, step-by-step learning path, recommended resources, and common mistakes to avoid. Format with bullet points.",
    "You are a senior software engineer. Explain {topic} with code examples, best practices, and common pitfalls. Structure your answer with clear sections.",
    "Provide a beginner-friendly introduction to {topic}. Cover: what it is, why it matters, how it works, and 3 practical use cases. Use numbered lists.",
    "Explain the concept of {topic} using the Feynman technique. Break it down into simple parts, use an analogy, and give one example. Limit to 200 words.",
    "Create a study guide for {topic} that includes: key terminology (5 terms), core concepts (3 points), practice questions (3 questions), and further reading suggestions.",
    "Write about {topic} from both a theoretical and practical perspective. Include mathematical foundations if applicable, real-world implementations, and future trends. Use structured headings.",
    "Explain {topic} in exactly 5 paragraphs: introduction, history, how it works, applications, and future scope. Each paragraph should be 3-4 sentences.",
    "You are a technical writer. Create documentation for {topic} including: overview, getting started guide, key features, and troubleshooting section. Use markdown formatting.",
    "Provide a critical analysis of {topic}. Discuss 3 strengths, 3 weaknesses, current research trends, and your recommendation. Support with examples.",
    "Design a lesson plan for teaching {topic} to beginners. Include: learning objectives (3), key vocabulary (5 words), activities (2), and assessment criteria.",
    "Explain {topic} with a focus on problem-solving. Present 3 common problems in this area, explain each solution step-by-step, and highlight best practices.",
    "Write a FAQ section for {topic} with 5 commonly asked questions and detailed answers. Each answer should be 2-3 sentences with examples where applicable.",
    "Summarize {topic} in a professional report format. Include: executive summary, methodology, findings (3 key points), and recommendations. Limit to 300 words.",
    "You are mentoring a junior developer. Explain {topic} with hands-on examples, common mistakes beginners make, and tips for mastering the concept. Use a friendly tone.",
    "Analyze how {topic} has evolved over the last decade. Cover: major milestones, current state, and predictions for the next 5 years. Use specific dates and examples.",
    "Create a cheat sheet for {topic}. Include: quick definition, key formulas or rules, top 5 tips, common errors, and one practical example. Keep it concise.",
    "Explain {topic} using the following structure: Problem it solves, How it works internally, When to use it, When NOT to use it, and one real project example.",
    "You are preparing exam notes. Explain {topic} covering all important aspects a student should know. Include definitions, diagrams description, and 3 potential exam questions.",
    "Write a blog post about {topic} for a tech audience. Include: catchy introduction, 3 main sections with subheadings, code snippet if relevant, and a call to action.",
]


def generate_dataset(num_samples=1000):
    """Generate a balanced prompt quality dataset."""
    prompts = []
    labels = []

    half = num_samples // 2

    # Generate low quality prompts
    for _ in range(half):
        topic = random.choice(topics)
        template = random.choice(low_quality_templates)
        prompt = template.format(topic=topic)

        # Randomly add minor variations
        if random.random() < 0.2:
            prompt = prompt.lower()
        if random.random() < 0.1:
            prompt = prompt.upper()

        prompts.append(prompt)
        labels.append(0)

    # Generate high quality prompts
    for _ in range(half):
        topic = random.choice(topics)
        template = random.choice(high_quality_templates)
        prompt = template.format(topic=topic)
        prompts.append(prompt)
        labels.append(1)

    # Create dataframe and shuffle
    df = pd.DataFrame({"prompt": prompts, "label": labels})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


def main():
    print("=" * 60)
    print("  PROMPT QUALITY DATASET GENERATOR")
    print("=" * 60)

    # Generate dataset
    num_samples = 1000
    print(f"\nGenerating {num_samples} samples...")
    df = generate_dataset(num_samples)

    # Save to CSV
    os.makedirs("dataset", exist_ok=True)
    output_path = os.path.join("dataset", "prompt_dataset.csv")
    df.to_csv(output_path, index=False)

    print(f"\nDataset saved to: {output_path}")
    print(f"Total samples: {len(df)}")
    print(f"Low Quality (0): {len(df[df['label'] == 0])}")
    print(f"High Quality (1): {len(df[df['label'] == 1])}")
    print(f"\nSample entries:")
    print(df.head(10).to_string(index=False))
    print("\n" + "=" * 60)
    print("  Dataset generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
