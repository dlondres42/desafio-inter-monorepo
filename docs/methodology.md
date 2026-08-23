# How would I approach this problem

This documentation describes my thought process during the coding of this case.

## Defining requirements

Before going ahead and coding the entire thing, or perhaps trying to one shot it with one prompt in claude, some careful planning is required to avoid later frustrations or even refactors. By reading the problem statement one can infer some functional requirements:

* A functional lib that abstracts some data science work (data exploration, training and inference). This lib should be published to pypi to later be consumed by external services, such as an API.
* An inference API deployed on K8s that consumes the abstraction described above and uses it to serve a model.
* Both components should have a robust CICD pipeline where they are properly conteinerized and served
* Documentation explaining how would I architect everything on AWS.

From the description alone it is necessary to decouple the lib from the API since they serve inherently different purposes. The API is concerned with only loading and serving the model and the lib needs to be a robust solution for the Data Scientist using it. For this reason, I opt to split the services in two different uv projects, each one with their unique configuration.

## Repository layout

Since the two deliverables are decoupled but graded as a single submission, the repository is a **monorepo with independent uv projects** behind a single remote, as the delivery requires one repository.

For CI I'm going with **GitHub Actions**. In a production environment I'd avoid creating a CI dependency on GHA, since GitHub is not really a great example of availability, but I'm following with the technology because it feels familiar to me and time is limited.

```
desafio-inter-monorepo/
├── README.md                 # front door: what this is, how to run it, index of everything
├── .github/
│   ├── workflows/            # one workflow per component, each with its own path filter
│   └── actions/              # composite actions shared across workflows (uv + cache setup)
│
├── data/                     # where the data is stored, namely: iris.csv as per required by the case
├── inference-server/         # fastapi APP
├── client_code/              # consumer proof: notebooks exercising the published lib
│
└── docs/                     # cross-cutting documentation
```

The `client_code` project is deliberately **not** a path dependency on `dolores-lib`. It installs the published artifact from test.pypi, which turns it into an end-to-end assertion that Stage 3 actually worked, if the wheel is
broken, the notebook fails.
