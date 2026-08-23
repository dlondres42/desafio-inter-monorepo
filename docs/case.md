# Desafio Técnico — Especialista MLOps

Referência rápida extraída de `~/Downloads/desafio_inter_mlops.pdf`.
Conteúdo mantido em português para não distorcer os requisitos originais.

> [!WARNING]
> **Prazo: 7 dias corridos a partir do recebimento do desafio.**
> O PDF foi baixado em **21/08/2026**, o que coloca a entrega em **28/08/2026**
> — confirme a data real de recebimento, essa é inferida do arquivo.

---

## Contexto

Vaga de **Especialista MLOps**: responsável por realizar deploys de modelos de
machine learning em produção, usando as práticas de CI/CD já estabelecidas na
empresa.

O desafio avalia competências técnicas em MLOps do desenvolvimento até a
implementação em ambiente produtivo. A instrução é completar **o máximo de
etapas possível** — não necessariamente todas.

---

## Etapas

### Etapa 1 — Desenvolvimento do pacote Python

- [ ] Pacote Python completo para **treinamento e inferência**, dataset **Iris**
- [ ] **Hatch** como sistema de build
- [ ] Boas práticas de desenvolvimento:
  - [ ] Design patterns adequados
  - [ ] Programação orientada a objetos (OOP)
  - [ ] Docstrings completas
  - [ ] Type hints
  - [ ] Código limpo e bem estruturado

### Etapa 2 — Integração Contínua (CI)

- [ ] Pipeline de CI no **GitLab.com**
- [ ] Testes automatizados:
  - [ ] **pytest** — testes unitários
  - [ ] **mypy** — verificação de tipos
  - [ ] **ruff** — linting
  - [ ] **black** — formatação

### Etapa 3 — Entrega Contínua (CD)

- [ ] Pipeline de CD publicando o pacote automaticamente no **test.pypi.org**
- [ ] Versionamento adequado do pacote

### Etapa 4 — Containerização

- [ ] **Dockerfile** otimizado que:
  - [ ] Instala o pacote das etapas anteriores
  - [ ] Configura uma **API de inferência** do modelo
  - [ ] Segue boas práticas de containerização

### Etapa 5 — Deploy Kubernetes

- [ ] Deploy da aplicação em **cluster Kubernetes local**
- [ ] Manifestos necessários (Deployment, Service, etc.)
- [ ] API de inferência acessível

### Etapa 6 — Arquitetura AWS (documentação)

- [ ] Desenhar e documentar arquitetura completa com:
  - [ ] **ECR** — Elastic Container Registry
  - [ ] **EKS** — Elastic Kubernetes Service
  - [ ] **SSM** — Systems Manager
  - [ ] **IAM** — Identity and Access Management
  - [ ] **VPC** — Virtual Private Cloud
- [ ] Explicar como **cada componente se integra** na solução

---

## Diferencial — Métricas de qualidade do modelo

Marcado como diferencial, mas aparece explicitamente nos critérios de
avaliação — vale tratar como parte do escopo, não como extra opcional.

- [ ] Sistema de análise de qualidade do modelo:
  - [ ] Accuracy
  - [ ] Precision
  - [ ] Recall
  - [ ] F1-Score
  - [ ] Confusion Matrix
  - [ ] Classification Report
- [ ] Exportar e **persistir** as métricas de forma estruturada (JSON, CSV ou banco)
- [ ] Documentar a **interpretação** das métricas no contexto do problema

---

## Critérios de avaliação

| Critério | Detalhe |
| --- | --- |
| Qualidade do código | Aderência às boas práticas |
| Funcionalidade | Pipelines de CI/CD funcionando |
| Completude | Da documentação |
| Criatividade | Na solução dos desafios |
| Conhecimento | Demonstrado em MLOps e infraestrutura |
| Métricas | Implementação das métricas de qualidade (diferencial) |

---

## Entrega

- [ ] **Repositório no GitLab.com** com todo o código desenvolvido
- [ ] **README detalhado** explicando como executar cada etapa
- [ ] **Documentação da arquitetura AWS** (diagramas + texto)
- [ ] **Relatório das métricas** de qualidade do modelo (se implementado)

---

