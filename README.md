# TradeOff_Especializacao_Generalizacao_LLMs
Trabalho 4 da disciplina de NLP 2025/1 - Análise Quantitativa do Trade-off entre Especialização e Generalização em LLMs via Fine-Tuning

## Para criar o ambiente de desenvolvimento pode se usar uma variação do comando abaixo

conda create --name trabalho_4_T2S_QA --file requirements.txt

## Execução dos scripts

Os scripts devem ser executados em ordem na ordem que eles estão dispostos no diretório 'scripts/'.

Existem mais de um com o mesmo número e isso significa que pode ser executado como script ou como Jupyter notebook que terá o mesmo resultado, essa opção fica com o objetivo de validar o passo a passo dos scripts mais longos de criação de prompts.

## Objetivo dos scripts:

- 01_Building_MMLU_dataset.py: realiza a criação da base de testes do MMLU que será utilizada para avaliação dos modelos treinados.
- 02_Building_spider_final_training_data.py: realiza a criação da base de dados de treinamento do Spider
- 03_Building_Spider_prompts.ipynb e 03_Building_Spider_prompts.py: contrói os prompts de treinamento e teste baseados nos arquivos de treinamento e desenvolvimento do Spider
- 04_FineTunning_LLaMa_text2sql.ipynb: realiza o treinamento do modelo LLaMa 3 8b com os hiper-parâmetros configurados, ao executar esse notebook para treinamento do modelo deve-se ter em mente a taxa de aprendizado para treinar a versão correta do modelo.
- 05_Test_LLaMA_text2sql.ipynb: Testa os modelos base e ajustados nos prompts de teste (dev split) do Spider
- 06_Running_sql_command.ipynb: Com os resultados dos prompts de teste do Spider deve-se executar esse script para montar o arquivo final de respostas casando com as informações anteriores da base de dados
- 07_Evaluating_query_answers.ipynb: Este script executa a avaliação dos modelos base e ajustados no prompts de teste do Spider (dev split).
- 08_Building_MMLU_prompts.ipynb e 08_Building_MMLU_prompts.py: Constrói os prompts de teste para a tarefa de resposta a perguntas do MMLU
- 09_Evaluating_LLaMA_MMLU.py: Avalia os modelos base e ajustados para os prompts de teste da tarefa de resposta a perguntas do MMLU

Obs: para executar os scripts python basta executar o comando:

python <nome_do_script>