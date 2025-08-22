# Introdução

Este projeto tem como objetivo explorar o desenvolvimento de um algoritmo de recomendação de vendas, uma ferramenta essencial para aprimorar a experiência do cliente e impulsionar o crescimento do negócio no setor de varejo. Em um mercado cada vez mais competitivo e personalizado, a capacidade de sugerir produtos relevantes aos consumidores, baseando-se em seus históricos de compra e no comportamento de outros usuários, torna-se um diferencial estratégico. Algoritmos de recomendação não apenas aumentam a probabilidade de vendas cruzadas e vendas adicionais, mas também contribuem para a fidelização do cliente, oferecendo uma jornada de compra mais intuitiva e satisfatória.

O trabalho se concentrará na aplicação de técnicas de aprendizado de máquina para construir um sistema que possa indicar novos produtos a um cliente com base em suas compras anteriores, e também sugerir itens que foram adquiridos por outros clientes com perfis de compra semelhantes. Para isso, utilizaremos como base conceitual o `Superstore Sales Dataset` do Kaggle, um conjunto de dados que, embora não seja manipulado diretamente para a execução de modelos neste estudo, oferece uma estrutura rica para a compreensão dos desafios e oportunidades inerentes à construção de um sistema de recomendação. A abordagem adotada neste documento detalhará a formulação do problema, os objetivos, a justificativa e o público-alvo, além de propor análises e visualizações que seriam fundamentais para o desenvolvimento de tal algoritmo.


## Problema

Nesta investigação, o problema central que buscamos resolver é a dificuldade em oferecer recomendações de produtos personalizadas e relevantes para os clientes de uma superloja. Em um cenário de vasto sortimento de produtos e um grande volume de transações, os clientes frequentemente enfrentam a sobrecarga de informação, tornando desafiador descobrir novos produtos que se alinhem aos seus interesses e necessidades. A ausência de um sistema de recomendação eficaz pode levar a:

*   **Perda de Oportunidades de Venda:** Clientes podem não estar cientes de produtos que lhes interessariam, resultando em vendas perdidas e menor valor de vida útil do cliente.
*   **Experiência do Cliente Subótima:** A dificuldade em encontrar produtos relevantes pode gerar frustração e diminuir o engajamento do cliente com a plataforma ou loja.
*   **Baixa Taxa de Conversão:** A falta de direcionamento pode levar a uma menor taxa de conversão de visitantes em compradores, ou de compradores existentes em compradores recorrentes.
*   **Ineficiência em Campanhas de Marketing:** Campanhas genéricas de marketing podem não ressoar com os interesses individuais dos clientes, resultando em baixo retorno sobre o investimento.

O contexto de aplicação para a solução deste problema é uma plataforma de e-commerce ou uma loja física com um sistema de vendas digitalizado, como a superloja global referenciada no dataset. A aplicação de um algoritmo de recomendação visa aprimorar a experiência de compra, sugerindo produtos de forma proativa. Especificamente, o algoritmo deverá ser capaz de:

1.  **Recomendar produtos novos com base no histórico de compras do próprio cliente:** Utilizando as informações de `Customer ID`, `Product ID`, `Order Date` e `Quantity`, o sistema deve identificar padrões de compra individuais e sugerir itens complementares ou de interesse similar.
2.  **Recomendar produtos com base no comportamento de compra de clientes semelhantes:** Analisando as compras de outros clientes (`Customer ID`) que adquiriram produtos similares (`Product ID`) ou que possuem padrões de compra correlatos, o algoritmo deve identificar tendências e sugerir produtos que foram populares entre esses grupos.


## Questão de Pesquisa

A questão de pesquisa é o pilar fundamental que orienta toda a investigação e análise de dados. Para o contexto do desenvolvimento de um algoritmo de recomendação de vendas utilizando o `Superstore Sales Dataset`, a questão de pesquisa que se busca responder é:

**"Como os dados históricos de compras de clientes, incluindo informações sobre produtos adquiridos e padrões de compra, podem ser utilizados para desenvolver um algoritmo de recomendação capaz de sugerir novos produtos a clientes individuais (baseado em histórico pessoal) e produtos populares entre clientes com perfis de compra semelhantes (baseado em comportamento coletivo)?"**

Esta questão de pesquisa é formulada para ser específica e objetiva, alinhando-se diretamente ao problema identificado da dificuldade em oferecer recomendações personalizadas. Ela direciona a investigação para a exploração dos dados históricos de transações (`Order ID`, `Customer ID`, `Product ID`, `Order Date`, `Quantity`, `Sales`), das características dos produtos (`Category`, `Sub-Category`, `Product Name`) e das informações de clientes (`Segment`). O foco é duplo: recomendações baseadas no histórico individual do cliente e recomendações baseadas na similaridade de comportamento entre clientes. Ao final do trabalho, espera-se que a análise das informações disponíveis e a proposição de metodologias permitam uma resposta fundamentada a esta questão, mesmo que de forma conceitual, dada a impossibilidade de acesso direto aos dados para execução de modelos.




## Objetivos Preliminares

Os objetivos preliminares deste trabalho são delineados para guiar a investigação sobre o desenvolvimento de um algoritmo de recomendação de vendas, com foco na experimentação de abordagens de aprendizado de máquina. É importante ressaltar que, nesta fase conceitual, os objetivos são formulados com base nas informações disponíveis sobre o dataset e na compreensão geral do problema de sistemas de recomendação.

**Objetivo Geral:**

Experimentar e avaliar modelos de aprendizado de máquina adequados para o desenvolvimento de um algoritmo de recomendação de vendas, utilizando as características de compra dos clientes e dos produtos presentes no dataset, visando otimizar a relevância e a precisão das recomendações.

**Objetivos Específicos:**

1.  **Explorar Padrões de Compra de Clientes:** Analisar o histórico de compras dos clientes (`Customer ID`, `Product ID`, `Order Date`, `Quantity`) para identificar padrões de consumo individuais e coletivos, como produtos frequentemente comprados juntos, sequência de compras e preferências por categorias ou sub-categorias de produtos.

2.  **Identificar Similaridades entre Clientes e Produtos:** Investigar métodos para determinar a similaridade entre clientes (com base em seus históricos de compra) e entre produtos (com base em suas características e como são comprados por diferentes clientes). Isso é fundamental para a aplicação de técnicas de filtragem colaborativa.

3.  **Propor Métricas de Avaliação e Estratégias de Validação para Sistemas de Recomendação:** Definir as métricas de avaliação de desempenho mais apropriadas para algoritmos de recomendação (e.g., Precisão, Recall, F1-score, Cobertura, Novidade) e propor estratégias de validação (e.g., divisão de dados em conjuntos de treinamento e teste, validação cruzada) que garantam a robustez e a generalização do algoritmo. Embora a execução não seja possível sem o dataset, a conceituação dessas métricas é crucial para a fase de modelagem.

Estes objetivos são preliminares e podem ser ajustados ou refinados à medida que a compreensão do dataset e do problema evoluir. Eles servem como um roteiro para a análise e a eventual construção de um algoritmo de recomendação de vendas eficaz.




## Justificativa

A escolha do `Superstore Sales Dataset` e a abordagem deste projeto para o desenvolvimento de um algoritmo de recomendação de vendas são justificadas pela crescente importância da personalização no varejo moderno. A capacidade de prever o que um cliente pode querer comprar a seguir, ou o que outros clientes com gostos semelhantes adquiriram, não é apenas uma conveniência para o consumidor, mas uma necessidade estratégica para as empresas.

As razões para a escolha dos objetivos específicos, que incluem a exploração de padrões de compra, a identificação de similaridades entre clientes e produtos, e a proposição de métricas de avaliação, residem na necessidade de construir um sistema de recomendação robusto e eficaz. A análise detalhada do histórico de compras permite desvendar preferências individuais e coletivas, enquanto a identificação de similaridades é a base para algoritmos de filtragem colaborativa, que são amplamente utilizados e bem-sucedidos. A definição de métricas apropriadas assegura que o desempenho do algoritmo seja avaliado de forma rigorosa, garantindo sua utilidade prática.

A relevância do estudo do problema dos sistemas de recomendação é inegável no cenário atual do comércio eletrônico e físico. Empresas como Amazon, Netflix e Spotify devem grande parte de seu sucesso à eficácia de seus algoritmos de recomendação. A implementação de um sistema de recomendação pode levar a um aumento significativo nas vendas (cross-selling e up-selling), na satisfação do cliente e na sua fidelização. Ao oferecer produtos relevantes, a experiência de compra se torna mais agradável e eficiente, reduzindo a sobrecarga de escolha para o consumidor e aumentando a probabilidade de conversão.

O impacto social, econômico e ambiental de um algoritmo de recomendação eficaz é considerável. Economicamente, a otimização das vendas e a melhoria da experiência do cliente resultam em maior receita e lucratividade para as empresas. Socialmente, a personalização das ofertas pode levar a uma maior satisfação do consumidor, que encontra produtos que realmente atendem às suas necessidades e desejos. 


## Público-Alvo

O público-alvo desta investigação e do eventual algoritmo de recomendação de vendas é diversificado, abrangendo diferentes perfis que podem se beneficiar diretamente da personalização das ofertas e da otimização da experiência de compra. Compreender esses perfis é fundamental para garantir que o sistema de recomendação seja relevante e eficaz.

Os principais perfis de pessoas ou grupos impactados incluem:

*   **Analistas de Dados e Cientistas de Dados:** Profissionais que atuam no desenvolvimento, implementação e otimização de sistemas de recomendação. Eles se beneficiarão da metodologia proposta para a análise de dados de transações, técnicas de modelagem de recomendação (como filtragem colaborativa e baseada em conteúdo) e avaliação de desempenho de algoritmos. O nível de familiaridade com recursos digitais e tecnologia é alto, e suas necessidades incluem a busca por soluções inovadoras e escaláveis para problemas de recomendação.

*   **Gerentes de Produto e Marketing:** Líderes e estrategistas que utilizam as recomendações para impulsionar vendas, aumentar o engajamento do cliente e personalizar campanhas. Para eles, o foco está na capacidade do algoritmo de gerar recomendações que resultem em maior conversão, valor médio do pedido e fidelização do cliente. O conhecimento prévio em tecnologia pode variar, mas a compreensão do comportamento do consumidor e das estratégias de mercado é crucial.

*   **Desenvolvedores de E-commerce e Plataformas Digitais:** Equipes responsáveis pela construção e manutenção de plataformas de vendas online. Eles se beneficiarão da compreensão dos requisitos técnicos e da arquitetura necessária para integrar um sistema de recomendação em suas plataformas, visando melhorar a usabilidade e a funcionalidade.

*   **Estudantes e Pesquisadores em Ciência de Dados e Inteligência Artificial:** Indivíduos em formação ou que buscam aprofundar seus conhecimentos em sistemas de recomendação. Este projeto pode servir como um estudo de caso prático, ilustrando as etapas desde a definição do problema até a proposição de soluções para um algoritmo de recomendação, utilizando um dataset real (ainda que conceitualmente). Eles buscam compreender as aplicações teóricas e práticas das técnicas de aprendizado de máquina em problemas de recomendação.

*   **Consumidores Finais:** O principal beneficiário do algoritmo de recomendação. Ao receber sugestões de produtos relevantes e personalizadas, o consumidor tem uma experiência de compra mais agradável, eficiente e satisfatória, descobrindo novos produtos que atendem às suas necessidades e preferências. Suas necessidades e expectativas estão ligadas à facilidade de encontrar o que procuram e à descoberta de itens que não conheciam, mas que lhes interessam.

O objetivo é compreender quem se beneficiaria da análise e dos resultados de um projeto de algoritmo de recomendação de vendas. As descrições são baseadas em perfis comuns no ambiente corporativo e acadêmico, garantindo que a comunicação e os resultados do projeto sejam direcionados de forma eficaz a cada grupo.



## Estado da Arte

Os sistemas de recomendação são uma área de pesquisa e aplicação em constante crescimento, impulsionada pela necessidade de auxiliar usuários a navegar por grandes volumes de informações e produtos. No contexto de vendas, esses sistemas visam sugerir itens que um cliente provavelmente comprará, com base em seu histórico e no comportamento de outros usuários. A literatura acadêmica e a indústria têm explorado diversas abordagens para construir sistemas de recomendação eficazes.

### Abordagens da Literatura

Os sistemas de recomendação podem ser amplamente categorizados em três tipos principais: filtragem colaborativa, filtragem baseada em conteúdo e abordagens híbridas.

#### Filtragem Colaborativa

A filtragem colaborativa é uma das abordagens mais populares e eficazes. Ela se baseia na ideia de que usuários que concordaram no passado continuarão a concordar no futuro. Existem dois subtipos principais:

*   **Baseada em Usuário (User-based):** Recomenda itens a um usuário com base nas preferências de usuários "semelhantes". A similaridade entre usuários é calculada com base em seus históricos de interação (e.g., compras, avaliações). Por exemplo, se o Usuário A e o Usuário B compraram muitos dos mesmos produtos, e o Usuário A comprou um produto X que o Usuário B ainda não comprou, o produto X pode ser recomendado ao Usuário B. [1]
*   **Baseada em Item (Item-based):** Recomenda itens que são "semelhantes" aos itens que o usuário já gostou ou comprou. A similaridade entre itens é calculada com base em como eles são avaliados ou comprados por diferentes usuários. Esta abordagem é frequentemente mais escalável para grandes bases de usuários. [1]

#### Filtragem Baseada em Conteúdo

Esta abordagem recomenda itens semelhantes aos que o usuário já demonstrou interesse no passado. A similaridade é calculada com base nas características dos itens e no perfil de preferência do usuário. Por exemplo, se um cliente compra frequentemente produtos da categoria "Tecnologia" e da sub-categoria "Telefones", o sistema pode recomendar outros telefones ou acessórios de tecnologia. [2]

#### Abordagens Híbridas

Sistemas híbridos combinam elementos da filtragem colaborativa e da filtragem baseada em conteúdo para superar as limitações de cada abordagem individualmente. Por exemplo, a filtragem colaborativa pode sofrer do problema de "cold start" (dificuldade em recomendar para novos usuários ou novos itens sem histórico de interação), enquanto a filtragem baseada em conteúdo pode ter dificuldade em descobrir itens fora do perfil de interesse explícito do usuário. Abordagens híbridas podem mitigar esses problemas, oferecendo recomendações mais robustas e diversificadas. [3]

### Métricas de Avaliação

A avaliação de sistemas de recomendação envolve métricas que medem a precisão, a cobertura, a novidade e a diversidade das recomendações. As métricas de precisão incluem:

*   **Precisão (Precision):** Proporção de itens recomendados que são relevantes.
*   **Recall:** Proporção de itens relevantes que foram recomendados.
*   **F1-score:** Média harmônica de precisão e recall.
*   **RMSE (Root Mean Squared Error) / MAE (Mean Absolute Error):** Usadas para avaliar a precisão de previsões de ratings em sistemas de recomendação baseados em ratings. [4]

Outras métricas importantes incluem:

*   **Cobertura (Coverage):** A proporção de itens ou usuários para os quais o sistema pode fazer recomendações.
*   **Novidade (Novelty):** A capacidade do sistema de recomendar itens que o usuário provavelmente não teria descoberto por conta própria.
*   **Diversidade (Diversity):** A variedade de itens recomendados ao usuário.

### Lacunas e Desafios

Apesar dos avanços, alguns desafios persistem no desenvolvimento de sistemas de recomendação:

*   **Problema do Cold Start:** Dificuldade em gerar recomendações para novos usuários (sem histórico de compras) ou novos itens (sem interações anteriores). Abordagens híbridas e baseadas em conteúdo podem ajudar a mitigar isso.
*   **Escalabilidade:** Para datasets muito grandes, como o de uma superloja global, a computação de similaridades e a geração de recomendações podem ser computacionalmente intensivas.
*   **Sparsity (Esparsidade):** A maioria dos usuários interage com apenas uma pequena fração dos itens disponíveis, resultando em matrizes de interação esparsas que dificultam a identificação de padrões.
*   **Dinâmica do Usuário e do Item:** As preferências dos usuários e a popularidade dos itens podem mudar ao longo do tempo, exigindo que os sistemas de recomendação sejam adaptáveis e atualizados continuamente.
*   **Interpretabilidade:** Entender por que um sistema de recomendação fez uma determinada sugestão pode ser complexo, especialmente com modelos de aprendizado profundo, o que pode ser um obstáculo para a confiança do usuário e a depuração do sistema.

### Alinhamento do Projeto com os Estudos Identificados

O presente projeto, ao focar no desenvolvimento de um algoritmo de recomendação de vendas para uma superloja global, alinha-se diretamente com as abordagens de filtragem colaborativa e baseada em conteúdo. A análise do `Superstore Sales Dataset`, mesmo que conceitual, permitirá explorar como os padrões de compra (`Order ID`, `Customer ID`, `Product ID`) e as características dos produtos (`Category`, `Sub-Category`, `Product Name`) podem ser utilizados para identificar similaridades e gerar recomendações. A questão de pesquisa proposta aborda explicitamente a recomendação baseada no histórico pessoal e no comportamento coletivo, o que remete diretamente às técnicas de filtragem colaborativa baseada em usuário e em item.

Este estudo servirá como um arcabouço para a compreensão dos desafios práticos, como o cold start e a escalabilidade, e para a proposição de soluções metodológicas. A ênfase na definição de métricas de avaliação apropriadas para sistemas de recomendação é crucial para garantir que, em uma futura implementação, o desempenho do algoritmo possa ser medido e otimizado de forma rigorosa. A pesquisa de estado da arte fornece a base teórica para a seleção de algoritmos e a compreensão das melhores práticas na construção de sistemas de recomendação eficazes.


# Descrição do Dataset Selecionado

Nesta seção, apresentamos uma visão clara e objetiva do dataset selecionado, o `Superstore Sales Dataset`, que servirá como base conceitual para o desenvolvimento do algoritmo de recomendação de vendas. Embora não tenhamos acesso direto aos dados para manipulação e execução de modelos, as informações descritivas fornecidas na página do Kaggle são suficientes para compreender a estrutura e o potencial do dataset para o problema de recomendação.

## Identificação e Origem

**Nome:** Superstore Sales Dataset

**Link de Acesso:** [https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting)

**Fonte:** Kaggle, disponibilizado por Rohit Sahoo.

**Licença de Uso:** GPL 2

## Visão Geral

Este dataset contém dados de vendas de uma superloja global ao longo de 4 anos. Originalmente projetado para análise de séries temporais e previsão de vendas, ele possui atributos ricos que são altamente relevantes para a construção de um sistema de recomendação. A presença de identificadores de cliente e produto, juntamente com informações de transação, permite a análise de padrões de compra e a identificação de similaridades entre usuários e itens.

**Total de Registros:** Não especificado diretamente, mas a prévia mostra 9800 linhas, indicando um volume considerável de transações.

**Total de Atributos:** 18 colunas, oferecendo uma variedade de informações sobre pedidos, clientes e produtos.

**Período Coberto:** 4 anos (datas exatas não especificadas, mas exemplos de 2015 a 2018 são mostrados), o que permite a análise de tendências de compra ao longo do tempo.

## Atributos

A tabela a seguir descreve os atributos disponíveis no dataset, com base nas informações e exemplos fornecidos na página do Kaggle. Destacamos a relevância de cada atributo para o contexto de um algoritmo de recomendação de vendas:

| Nome do Atributo | Descrição | Tipo | Relevância para Recomendação | Exemplos de Valores |
|---|---|---|---|---|
| Row ID | Identificador único para cada linha/registro. | Numérico | Identificação de transações individuais. | 1, 2, 3, ... |
| Order ID | Identificador único para cada pedido. | Alfanumérico | Agrupamento de produtos comprados em uma mesma transação. | CA-2017-152156, US-2016-108966 |
| Order Date | Data em que o pedido foi realizado. | Data | Análise de temporalidade das compras e tendências. | 08/11/2017, 12/06/2017 |
| Ship Date | Data em que o pedido foi enviado. | Data | Informação secundária para o histórico de compra. | 11/11/2017, 16/06/2017 |
| Ship Mode | Modo de envio do pedido. | Categórico | Pode indicar preferências de serviço do cliente. | Second Class, Standard Class, First Class, Same Day |
| Customer ID | Identificador único do cliente. | Alfanumérico | **Crucial** para identificar o histórico de compra de cada usuário e para a filtragem colaborativa baseada em usuário. | CG-12520, DV-13045 |
| Customer Name | Nome do cliente. | Texto | Identificação do cliente, mas não diretamente usado em algoritmos. | Claire Gute, Darrin Van Huff |
| Segment | Segmento de cliente. | Categórico | Pode ser usado para segmentar recomendações ou para filtragem baseada em conteúdo de usuário. | Consumer, Corporate, Home Office |
| Country | País do cliente. | Texto | Análise geográfica de padrões de compra. | United States |
| City | Cidade do cliente. | Texto | Análise geográfica de padrões de compra. | Henderson, Los Angeles, Fort Lauderdale |
| State | Estado do cliente. | Texto | Análise geográfica de padrões de compra. | California, Texas, New York |
| Postal Code | Código postal do cliente. | Numérico | Análise geográfica mais granular. | 42420, 90036 |
| Region | Região de vendas. | Categórico | Análise geográfica de padrões de compra. | South, West, East, Central |
| Product ID | Identificador único do produto. | Alfanumérico | **Crucial** para identificar os produtos comprados e para a filtragem colaborativa baseada em item. | FUR-BO-10001798, TEC-PH-10002075 |
| Category | Categoria do produto. | Categórico | **Crucial** para filtragem baseada em conteúdo de item e para agrupar produtos. | Furniture, Office Supplies, Technology |
| Sub-Category | Sub-categoria do produto. | Categórico | **Crucial** para filtragem baseada em conteúdo de item e para agrupar produtos. | Bookcases, Chairs, Phones, Paper |
| Product Name | Nome do produto. | Texto | Identificação do produto, pode ser usado para extração de características textuais. | Bush Somerset Collection Bookcase, Hon Deluxe Fabric Upholstered Stacking Chairs |
| Sales | Valor total das vendas. | Numérico | Pode indicar a importância ou o valor de um item para o cliente. | 261.96, 731.94 |
| Quantity | Quantidade de itens vendidos. | Numérico | Pode indicar a frequência de compra ou a demanda por um item. | 2, 3, 5 |
| Discount | Valor do desconto aplicado. | Numérico | Pode influenciar o comportamento de compra e a relevância da recomendação. | 0.00, 0.20 |
| Profit | Lucro gerado pela venda. | Numérico | Informação para análise de negócio, não diretamente para recomendação. | 41.94, 219.58 |

## Qualidade dos Dados

A página do Kaggle afirma que o dataset é "fácil de entender e autoexplicativo". No entanto, não há informações explícitas sobre a presença de valores faltantes, inconsistências, duplicatas ou outliers. Para a construção de um algoritmo de recomendação, a qualidade dos dados é fundamental. A presença de IDs de cliente e produto únicos e consistentes é crucial. Uma análise exploratória de dados (EDA) seria necessária para identificar e tratar esses problemas de qualidade de dados, caso existam, garantindo a integridade e a confiabilidade das recomendações geradas. A seção de "Usability" do dataset no Kaggle indica uma pontuação de 10.00, o que sugere uma boa qualidade geral, mas detalhes específicos sobre anomalias não são fornecidos, o que exigiria uma verificação rigorosa em um projeto com acesso direto aos dados.


# Canvas analítico

Nesta seção, você deverá estruturar e preencher o seu Canvas Analítico, que tem como objetivo registrar a organização das ideias e apresentar o modelo de negócio do projeto.

O Canvas deve ser preenchido integralmente, mesmo que algumas informações ainda não estejam totalmente definidas. Nessa etapa inicial, é aceitável trabalhar com hipóteses ou estimativas, desde que sejam coerentes com o problema e o contexto definidos.

**Dica:** O Canvas Analítico serve como guia visual para alinhar expectativas e direcionar o desenvolvimento. Ele poderá (e deverá) ser revisitado e atualizado ao longo do projeto.

> **Links Úteis**:
> - [Modelo do Canvas Analítico](https://github.com/ICEI-PUC-Minas-PMV-SI/PesquisaExperimentacao-Template/blob/main/help/Software-Analtics-Canvas-v1.0.pdf)

# Vídeo de apresentação da Etapa 01

Nesta etapa, o grupo deverá produzir um vídeo de 5 a 8 minutos apresentando o trabalho realizado, no qual cada integrante deve dizer seu nome e apresentar uma parte do conteúdo desenvolvido, garantindo que todos participem ativamente da gravação. A ausência de participação de qualquer membro resultará em penalização na nota final desta etapa. Recomenda-se que o grupo elabore previamente um roteiro para organizar a ordem das falas, distribuir o tempo de forma equilibrada e assegurar que todos os tópicos relevantes sejam apresentados de maneira clara e objetiva.

# Referências

Inclua todas as referências (livros, artigos, sites, etc) utilizados no desenvolvimento do trabalho utilizando o padrão ABNT.

> **Links Úteis**:
> - [Padrão ABNT PUC Minas](https://portal.pucminas.br/biblioteca/index_padrao.php?pagina=5886)
