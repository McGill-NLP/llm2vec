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
        - label: "Blog"
          url: "https://www.servicenow.com"
          icon: "fas fa-newspaper"
        - label: "Tutorial"
          url: "https://mcgill-nlp.github.io/llm2vec/tutorial"
          icon: "fas fa-laptop"
        - label: "Models"
          url: "https://huggingface.co/collections/McGill-NLP/llm2vec-660e14f536b3c8d10b3f1c34"
          icon: "fas fa-robot"
        - label: "Code"
          url: "https://github.com/McGill-NLP/llm2vec"
          icon: "fab fa-github"

title: "LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders"
excerpt: Parishad BehnamGhader\*, Vaibhav Adlakha\*, Marius Mosbach, Dzmitry Bahdanau, Nicolas Chapados, Siva Reddy
---

LLM2Vec is a simple recipe to convert decoder-only LLMs into text encoders. It consists of 3 simple steps: 1) enabling bidirectional attention, 2) training with masked next token prediction, and 3) unsupervised contrastive learning. Please take a look at the links above for more information.

<p align="center">
  <img src="https://github.com/McGill-NLP/llm2vec/assets/12207571/48efd48a-431b-4625-8e0f-248a442e3839" width="75%" alt="LLM2Vec_figure1"/>
</p>
