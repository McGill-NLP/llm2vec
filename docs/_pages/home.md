---
permalink: /
layout: splash
header:
    # overlay_color: rgb(237, 27, 47)
    overlay_color: rgb(180, 27, 47)
    actions:
        - label: "Paper"
          url: https://arxiv.org
          icon: "fas fa-book"
        - label: "Code"
          url: "https://github.com/McGill-NLP/llm2vec"
          icon: "fab fa-github"
        - label: "Tutorial"
          url: "https://mcgill-nlp.github.io/llm2vec/tutorial"
          icon: "fas fa-laptop"
        - label: "Models"
          url: "https://huggingface.co/collections/McGill-NLP/llm2vec-660e14f536b3c8d10b3f1c34"
          icon: "fas fa-robot"
        - label: "PR?"
          url: "https://www.servicenow.com"
          icon: "fas fa-newspaper"

title: "LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders"
excerpt: Parishad BehnamGhader\*, Vaibhav Adlakha\*, Marius Mosbach, Dzmitry Bahdanau, Nicolas Chapados, Siva Reddy
---

LLM2Vec is a simple recipe to convert decoder-only LLMs into text encoders. It consists of 3 simple steps: 1) enabling bidirectional attention, 2) training with masked next token prediction, and 3) unsupervised contrastive learning. The model can be further fine-tuned to achieve state-of-the-art performance.

<p align="center">
  <img src="https://github.com/McGill-NLP/llm2vec/assets/12207571/48efd48a-431b-4625-8e0f-248a442e3839" width="75%" alt="LLM2Vec_figure1"/>
</p>

Please take a look at [our GitHub repository](https://github.com/McGill-NLP/llm2vec) for a guide to how to install and use LLM2Vec.
