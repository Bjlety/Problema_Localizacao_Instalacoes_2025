import pandas as pd
import numpy as np
import itertools
import time
import datetime
import os
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Constantes para os idiomas
IDIOMAS = {
    "pt": {
        "pacientes": "pacientes.csv",
        "hospitais": "hospitais.csv",
        "cabecalho_pacientes": "Bairro dos Pacientes;Coordenada x;Coordenada y;Demanda",
        "cabecalho_hospitais": "Local do Hospital;Coordenada x;Coordenada y;Capacidade;Custo",
        "escolha_idioma": "Escolha o idioma:",
        "opcao_importar": "Você deseja importar os dados dos arquivos CSV (1) ou inserir novos dados (2)?",
        "opcao_importar_arquivo_nao_existe": "Os arquivos CSV não existem. Inserindo novos dados...",
        "criando_arquivo": "Criando arquivo ",
        "inserir_pacientes": "Insira os dados dos pacientes (digite 'fim' para terminar):",
        "inserir_hospitais": "Insira os dados dos hospitais (digite 'fim' para terminar):",
        "nome_paciente": "Bairro dos Pacientes ",
        "coord_x": "Coordenada x ",
        "coord_y": "Coordenada y ",
        "demanda": "Demanda ",
        "local_hospital": "Local do Hospital ",
        "capacidade": "Capacidade ",
        "custo": "Custo ",
        "erro_importar": "Erro ao importar os dados. Verifique os arquivos CSV.",
        "erro_p_medianas": "O número de hospitais deve estar entre 1 e o número de instalações.",
        "num_p_medianas": "Digite o número de hospitais (p): ",
        "fim_entrada_dados": "Fim da entrada de dados.",
        "erro_cabecalho_invalido": "Cabeçalho do arquivo inválido. Corrigindo...",
        "erro_num_p_maior_hospitais": "O número p de medianas não pode ser maior do que a quantidade de hospitais disponíveis",
        "erro_num_p_menor_um": "O número p de medianas não pode ser menor do que 1",
        "erro_pacientes_menor_zero": "O número de pacientes deve ser maior que 0",
        "calculando_solucao": "Calculando a solução ótima... Isso pode levar até 5 minutos.",
        "solucao_otima_encontrada": "Solução ótima encontrada:",
        "solucao_otima_nao_encontrada": "Solução ótima não encontrada dentro do limite de tempo. Exibindo a melhor solução encontrada.",
        "distancia_media": "Distância média: ",
        "custo_total": "Custo total: ",
        "hospitais_a": "Hospitais:", 
        "pacientes_atendidos": "Pacientes atendidos por cada hospital:",
        "refazer_operacao": "Você deseja refazer a operação (1) ou encerrar o programa (2)?",
        "encerrando_programa": "Encerrando o programa...",
        "km": " km",
        "pessoas": " pessoas",
        "dinheiro": " reais",
        "data_geracao": "Data de geração: ",
        "grafico_titulo": "Alocação de Pacientes e Hospitais",
        "grafico_legenda": "Legenda:",
        "grafico_pacientes": "Pacientes",
        "grafico_hospitais": "Hospitais",
        "melhor_solucao_titulo": "Melhor Solução Encontrada",
        "melhor_solucao_metodo": "Método",
        "melhor_solucao_distancia": "Distância Média",
        "melhor_solucao_custo": "Custo Total",
        "melhor_solucao_explicacao_fb": "A solução de Força Bruta é considerada a melhor, pois explora todas as combinações possíveis e garante a solução ótima global.",
        "melhor_solucao_explicacao_g": "A solução Gulosa é considerada a melhor neste caso. Embora seja uma abordagem heurística, ela encontrou a solução com menor custo e distância média entre as testadas, sugerindo uma boa aproximação da solução ótima.",
        "melhor_solucao_explicacao_pd": "A solução de Programação Dinâmica é considerada a melhor. Este método utiliza uma abordagem mais sistemática para explorar o espaço de soluções, dividindo o problema em subproblemas e armazenando suas soluções para evitar recálculos, o que pode levar a resultados mais precisos do que abordagens puramente gulosas, especialmente em problemas de otimização combinatória.",
        "melhor_solucao_explicacao_dc": "A solução de Divisão e Conquista é considerada a melhor. Este método divide o problema original em subproblemas menores, resolve esses subproblemas recursivamente e depois combina suas soluções para obter a solução do problema original. Embora esta abordagem possa não garantir a otimalidade global como a Força Bruta, ela demonstrou ser eficaz neste caso, encontrando uma solução com custo e distância média competitivos em relação aos outros métodos.",
        "solucao_z": "Solução",
        "metodo": "Método",
        "tempo": "Tempo de Execução",
        "resultado_otimizacao": "Resultado da Otimização",
        "relatorio_otimizacao": "Relatório de Otimização",
        "problema_p_medianas": "Problema das p-Medianas Capacitadas",
        "hospital": "Hospital",
        "nao_utilizado": "Não Utilizado",
        "arquivo_nao_encontrado": "Arquivo {nome_arquivo} não encontrado. Criando um novo arquivo.",
        "arquivo_vazio": "Arquivo {nome_arquivo} está vazio. Insira os dados.",
        "erro_colunas_faltantes": "O DataFrame de {nome_dataframe} não possui as colunas necessárias: {lista_colunas}. Cabeçalho esperado: {cabecalho_esperado}",
        "erro_hospitais_nao_inicializado": "Erro: O DataFrame de hospitais não foi inicializado corretamente. Tente novamente ou insira os dados manualmente.",
        "fim": "fim",
        "introducao": """
-------------------------------------------------------------------------------
Bem-vindo ao Programa de Resolução do Problema das p-Medianas Capacitadas!

Este programa foi desenvolvido para encontrar a melhor alocação de hospitais (medianas)
para atender a um conjunto de pacientes, considerando a capacidade de cada hospital e os custos envolvidos.

**Problema das p-Medianas Capacitadas:**

O problema das p-medianas capacitadas consiste em, dado um conjunto de pacientes com demandas específicas e
um conjunto de possíveis locais para instalação de hospitais, cada um com uma capacidade e um custo de instalação,
encontrar os *p* locais que minimizam a soma das distâncias ponderadas entre os pacientes e os hospitais aos quais estão alocados,
respeitando a capacidade de cada hospital.

**Como o programa funciona:**

1. **Entrada de Dados:**
    - Você pode importar os dados dos pacientes e dos hospitais a partir de arquivos CSV (`pacientes.csv` e `hospitais.csv`)
      ou inserir os dados manualmente.
    - Os arquivos CSV devem ter um cabeçalho específico, conforme detalhado nas mensagens de erro.
    - O programa irá guiá-lo na inserção de dados, caso os arquivos não existam, estejam vazios ou com cabeçalhos incorretos.
    - Você também precisará informar o número *p* de hospitais que deseja instalar.

2. **Métodos de Resolução:**
    - O programa utiliza quatro métodos para resolver o problema:
      - **Força Bruta:** Testa todas as combinações possíveis de *p* hospitais (método exato, porém computacionalmente caro para instâncias grandes).
      - **Guloso:** Constrói uma solução iterativamente, escolhendo a cada passo o hospital que mais reduz a distância total ponderada (método heurístico, rápido mas pode não encontrar a solução ótima).
      - **Programação Dinâmica:** Divide o problema em subproblemas e armazena as soluções para evitar recálculos (método mais eficiente que a força bruta para certos problemas, mas pode ter limitações de memória).
      - **Divisão e Conquista:** Divide o problema geograficamente em subproblemas menores e resolve-os recursivamente, combinando as soluções (método heurístico, com potencial para ser rápido).

3. **Saída:**
    - Para cada método, o programa gera um arquivo HTML com os resultados detalhados, incluindo:
      - Lista dos hospitais escolhidos, com suas coordenadas e custo.
      - Lista de pacientes atendidos por cada hospital.
      - Distância média entre os pacientes e os hospitais aos quais estão alocados.
      - Custo total da solução.
      - Tempo de execução do método.
      - Um gráfico mostrando a alocação dos pacientes aos hospitais.
    - O programa também identifica a melhor solução encontrada entre todos os métodos, considerando o menor custo total e a menor distância média.

**Observações:**

- O tempo de execução pode variar dependendo do tamanho da instância e do método escolhido.
- O método de força bruta pode demorar um tempo considerável para instâncias grandes.
- Os arquivos HTML e os gráficos são gerados na mesma pasta onde o programa está localizado.

-------------------------------------------------------------------------------
""",
    },
    "es": {
        "pacientes": "pacientes.csv",
        "hospitais": "hospitales.csv",
        "cabecalho_pacientes": "Barrio de los Pacientes;Coordenada x;Coordenada y;Demanda",
        "cabecalho_hospitais": "Ubicación del Hospital;Coordenada x;Coordenada y;Capacidad;Costo",
        "escolha_idioma": "Seleccione el idioma:",
        "opcao_importar": "¿Desea importar los datos de los archivos CSV (1) o ingresar nuevos datos (2)?",
        "opcao_importar_arquivo_nao_existe": "Los archivos CSV no existen. Ingresando nuevos datos...",
        "criando_arquivo": "Creando archivo ",
        "inserir_pacientes": "Ingrese los datos de los pacientes (escriba 'fin' para terminar):",
        "inserir_hospitais": "Ingrese los datos de los hospitales (escriba 'fin' para terminar):",
        "nome_paciente": "Barrio de los Pacientes ",
        "coord_x": "Coordenada x ",
        "coord_y": "Coordenada y ",
        "demanda": "Demanda ",
        "local_hospital": "Ubicación del Hospital ",
        "capacidade": "Capacidad ",
        "custo": "Costo ",
        "erro_importar": "Error al importar los datos. Verifique los archivos CSV.",
        "erro_p_medianas": "El número de hospitales debe estar entre 1 y el número de instalaciones.",
        "num_p_medianas": "Ingrese el número de hospitales (p): ",
        "fim_entrada_dados": "Fin de la entrada de datos.",
        "erro_cabecalho_invalido": "Encabezado del archivo inválido. Corrigiendo...",
        "erro_num_p_maior_hospitais": "El número p de medianas no puede ser mayor que la cantidad de hospitales disponibles",
        "erro_num_p_menor_um": "El número p de medianas no puede ser menor que 1",
        "erro_pacientes_menor_zero": "El número de pacientes debe ser mayor que 0",
        "calculando_solucao": "Calculando la solución óptima... Esto puede tomar hasta 5 minutos.",
        "solucao_otima_encontrada": "Solución óptima encontrada:",
        "solucao_otima_nao_encontrada": "Solución óptima no encontrada dentro del límite de tiempo. Mostrando la mejor solución encontrada.",
        "distancia_media": "Distancia media: ",
        "custo_total": "Costo total: ",
        "hospitais_a": "Hospitales:",
        "pacientes_atendidos": "Pacientes atendidos por cada hospital:",
        "refazer_operacao": "¿Desea rehacer la operación (1) o cerrar el programa (2)?",
        "encerrando_programa": "Cerrando el programa...",
        "km": " km",
        "pessoas": " personas",
        "dinheiro": " reais",
        "data_geracao": "Fecha de generación: ",
        "grafico_titulo": "Asignación de Pacientes y Hospitales",
        "grafico_legenda": "Leyenda:",
        "grafico_pacientes": "Pacientes",
        "grafico_hospitais": "Hospitales",
        "melhor_solucao_titulo": "Mejor Solución Encontrada",
        "melhor_solucao_metodo": "Método",
        "melhor_solucao_distancia": "Distancia Promedio",
        "melhor_solucao_custo": "Costo Total",
        "melhor_solucao_explicacao_fb": "La solución de Fuerza Bruta se considera la mejor, ya que explora todas las combinaciones posibles y garantiza la solución óptima global.",
        "melhor_solucao_explicacao_g": "La solución Golosa se considera la mejor en este caso. Aunque es un enfoque heurístico, encontró la solución con el menor costo y la menor distancia promedio entre las probadas, lo que sugiere una buena aproximación a la solución óptima.",
        "melhor_solucao_explicacao_pd": "La solución de Programación Dinámica se considera la mejor. Este método utiliza un enfoque más sistemático para explorar el espacio de soluciones, dividiendo el problema en subproblemas y almacenando sus soluciones para evitar recálculos, lo que puede conducir a resultados más precisos que los enfoques puramente codiciosos, especialmente en problemas de optimización combinatoria.",
        "melhor_solucao_explicacao_dc": "La solución de División y Conquista se considera la mejor. Este método divide el problema original en subproblemas más pequeños, resuelve estos subproblemas de forma recursiva y luego combina sus soluciones para obtener la solución del problema original. Aunque este enfoque puede no garantizar la optimalidad global como la Fuerza Bruta, demostró ser efectivo en este caso, encontrando una solución con costo y distancia promedio competitivos en relación con los otros métodos.",
        "solucao_z": "Solución",
        "metodo": "Método",
        "tempo": "Tiempo de Ejecución",
        "resultado_otimizacao": "Resultado de la Optimización",
        "relatorio_otimizacao": "Informe de Optimización",
        "problema_p_medianas": "Problema de las p-Medianas Capacitadas",
        "hospital": "Hospital",
        "nao_utilizado": "No Utilizado",
        "erro_cabecalho_invalido": "Encabezado del archivo inválido. Corrigiendo...",
        "arquivo_nao_encontrado": "Archivo {nome_arquivo} no encontrado. Creando un nuevo archivo.",
        "arquivo_vazio": "Archivo {nome_arquivo} está vacío. Ingrese los datos.",
        "erro_colunas_faltantes": "El DataFrame de {nome_dataframe} no posee las columnas necesarias: {lista_colunas}. Encabezado esperado: {cabecalho_esperado}",
        "erro_hospitais_nao_inicializado": "Error: El DataFrame de hospitales no fue inicializado correctamente. Intente nuevamente o ingrese los datos manualmente.",
        "fim": "fin",
        "introducao": """
-------------------------------------------------------------------------------
¡Bienvenido al Programa de Resolución del Problema de las p-Medianas Capacitadas!

Este programa fue desarrollado para encontrar la mejor asignación de hospitales (medianas)
para atender a un conjunto de pacientes, considerando la capacidad de cada hospital y los costos involucrados.

**Problema de las p-Medianas Capacitadas:**

El problema de las p-medianas capacitadas consiste en, dado un conjunto de pacientes con demandas específicas y
un conjunto de posibles ubicaciones para la instalación de hospitales, cada uno con una capacidad y un costo de instalación,
encontrar las *p* ubicaciones que minimizan la suma de las distancias ponderadas entre los pacientes y los hospitales a los que están asignados,
respetando la capacidad de cada hospital.

**Cómo funciona el programa:**

1. **Entrada de Datos:**
    - Puede importar los datos de los pacientes y los hospitales desde archivos CSV (`pacientes.csv` e `hospitales.csv`)
      o ingresar los datos manualmente.
    - Los archivos CSV deben tener un encabezado específico, como se detalla en los mensajes de error.
    - El programa lo guiará en la inserción de datos, en caso de que los archivos no existan, estén vacíos o con encabezados incorrectos.
    - También deberá ingresar el número *p* de hospitales que desea instalar.

2. **Métodos de Resolución:**
    - El programa utiliza cuatro métodos para resolver el problema:
      - **Fuerza Bruta:** Prueba todas las combinaciones posibles de *p* hospitales (método exacto, pero computacionalmente caro para instancias grandes).
      - **Goloso:** Construye una solución iterativamente, eligiendo en cada paso el hospital que más reduce la distancia total ponderada (método heurístico, rápido pero puede no encontrar la solución óptima).
      - **Programación Dinámica:** Divide el problema en subproblemas y almacena las soluciones para evitar recálculos (método más eficiente que la fuerza bruta para ciertos problemas, pero puede tener limitaciones de memoria).
      - **División y Conquista:** Divide el problema geográficamente en subproblemas más pequeños y los resuelve recursivamente, combinando las soluciones (método heurístico, con potencial para ser rápido).

3. **Salida:**
    - Para cada método, el programa genera un archivo HTML con los resultados detallados, incluyendo:
      - Lista de los hospitales elegidos, con sus coordenadas y costo.
      - Lista de pacientes atendidos por cada hospital.
      - Distancia promedio entre los pacientes y los hospitales a los que están asignados.
      - Costo total de la solución.
      - Tiempo de ejecución del método.
      - Un gráfico que muestra la asignación de los pacientes a los hospitales.
    - El programa también identifica la mejor solución encontrada entre todos los métodos, considerando el menor costo total y la menor distancia promedio.

**Observaciones:**

- El tiempo de ejecución puede variar dependiendo del tamaño de la instancia y del método elegido.
- El método de fuerza bruta puede tardar un tiempo considerable para instancias grandes.
- Los archivos HTML y los gráficos se generan en la misma carpeta donde se encuentra el programa.

-------------------------------------------------------------------------------
""",
    },
    "en": {
        "pacientes": "patients.csv",
        "hospitais": "hospitals.csv",
        "cabecalho_pacientes": "Patients Quarter;Coordinate x;Coordinate y;Demand",
        "cabecalho_hospitais": "Hospital Location;Coordinate x;Coordinate y;Capacity;Cost",
        "escolha_idioma": "Choose the language:",
        "opcao_importar": "Do you want to import data from CSV files (1) or enter new data (2)?",
        "opcao_importar_arquivo_nao_existe": "CSV files do not exist. Entering new data...",
        "criando_arquivo": "Creating file ",
        "inserir_pacientes": "Enter patient data (type 'end' to finish):",
        "inserir_hospitais": "Enter hospital data (type 'end' to finish):",
        "nome_paciente": "Patients Quarter ",
        "coord_x": "Coordinate x ",
        "coord_y": "Coordinate y ",
        "demanda": "Demand ",
        "local_hospital": "Hospital Location ",
        "capacidade": "Capacity ",
        "custo": "Cost ",
        "erro_importar": "Error importing data. Check the CSV files.",
        "erro_p_medianas": "The number of hospitals must be between 1 and the number of facilities.",
        "num_p_medianas": "Enter the number of hospitals (p): ",
        "fim_entrada_dados": "End of data entry.",
        "erro_cabecalho_invalido": "Invalid file header. Correcting...",
        "erro_num_p_maior_hospitais": "The number p of medians cannot be greater than the number of hospitals available",
        "erro_num_p_menor_um": "The number p of medians cannot be less than 1",
        "erro_pacientes_menor_zero": "The number of patients must be greater than 0",
        "calculando_solucao": "Calculating the optimal solution... This may take up to 5 minutes.",
        "solucao_otima_encontrada": "Optimal solution found:",
        "solucao_otima_nao_encontrada": "Optimal solution not found within the time limit. Displaying the best solution found.",
        "distancia_media": "Average distance: ",
        "custo_total": "Total cost: ",
        "hospitais_a": "Hospitals:",
        "pacientes_atendidos": "Patients served by each hospital:",
        "refazer_operacao": "Do you want to redo the operation (1) or close the program (2)?",
        "encerrando_programa": "Closing the program...",
        "km": " km",
        "pessoas": " people",
        "dinheiro": " reais",
        "data_geracao": "Generation date: ",
        "grafico_titulo": "Allocation of Patients and Hospitals",
        "grafico_legenda": "Legend:",
        "grafico_pacientes": "Patients",
        "grafico_hospitais": "Hospitals",
        "melhor_solucao_titulo": "Best Solution Found",
        "melhor_solucao_metodo": "Method",
        "melhor_solucao_distancia": "Average Distance",
        "melhor_solucao_custo": "Total Cost",
        "melhor_solucao_explicacao_fb": "The Brute Force solution is considered the best, as it explores all possible combinations and guarantees the global optimal solution.",
        "melhor_solucao_explicacao_g": "The Greedy solution is considered the best in this case. Although it is a heuristic approach, it found the solution with the lowest cost and average distance among those tested, suggesting a good approximation of the optimal solution.",
        "melhor_solucao_explicacao_pd": "The Dynamic Programming solution is considered the best. This method uses a more systematic approach to exploring the solution space, breaking the problem down into subproblems and storing their solutions to avoid recalculations, which can lead to more accurate results than purely greedy approaches, especially in combinatorial optimization problems.",
        "melhor_solucao_explicacao_dc": "The Divide and Conquer solution is considered the best. This method divides the original problem into smaller subproblems, solves these subproblems recursively, and then combines their solutions to obtain the solution to the original problem. Although this approach may not guarantee global optimality like Brute Force, it proved to be effective in this case, finding a solution with competitive cost and average distance compared to the other methods.",
        "solucao_z": "Solution",
        "metodo": "Method",
        "tempo": "Execution Time",
        "resultado_otimizacao": "Optimization Result",
        "relatorio_otimizacao": "Optimization Report",
        "problema_p_medianas": "Capacitated p-Median Problem",
        "hospital": "Hospital",
        "nao_utilizado": "Not Used",
        "erro_cabecalho_invalido": "Invalid file header. Correcting...",
        "arquivo_nao_encontrado": "File {nome_arquivo} not found. Creating a new file.",
        "arquivo_vazio": "File {nome_arquivo} is empty. Enter the data.",
        "erro_colunas_faltantes": "The DataFrame of {nome_dataframe} does not have the necessary columns: {lista_colunas}. Expected header: {cabecalho_esperado}",
        "erro_hospitais_nao_inicializado": "Error: The hospitals DataFrame was not initialized correctly. Try again or enter the data manually.",
        "fim": "end",
        "introducao": """
-------------------------------------------------------------------------------
Welcome to the Capacitated p-Median Problem Solver Program!

This program was developed to find the best allocation of hospitals (medians)
to serve a set of patients, considering the capacity of each hospital and the costs involved.

**Capacitated p-Median Problem:**

The capacitated p-median problem consists of, given a set of patients with specific demands and
a set of possible locations for the installation of hospitals, each with a capacity and an installation cost,
finding the *p* locations that minimize the sum of the weighted distances between patients and the hospitals to which they are allocated,
respecting the capacity of each hospital.

**How the program works:**

1. **Entrada de Dados:**
    - You can import patient and hospital data from CSV files (`patients.csv` and `hospitals.csv`)
      or enter the data manually.
    - The CSV files must have a specific header, as detailed in the error messages.
    - The program will guide you in entering data, in case the files do not exist, are empty or have incorrect headers.
    - You will also need to enter the number *p* of hospitals you want to install.

2. **Solution Methods:**
    - The program uses four methods to solve the problem:
      - **Brute Force:** Tests all possible combinations of *p* hospitals (exact method, but computationally expensive for large instances).
      - **Greedy:** Builds a solution iteratively, choosing at each step the hospital that most reduces the total weighted distance (heuristic method, fast but may not find the optimal solution).
      - **Dynamic Programming:** Divides the problem into subproblems and stores the solutions to avoid recalculations (more efficient method than brute force for certain problems, but may have memory limitations).
      - **Divide and Conquer:** Divides the problem geographically into smaller subproblems and solves them recursively, combining the solutions (heuristic method, with potential to be fast).

3. **Output:**
    - For each method, the program generates an HTML file with the detailed results, including:
      - List of chosen hospitals, with their coordinates and cost.
      - List of patients served by each hospital.
      - Average distance between patients and the hospitals to which they are allocated.
      - Total cost of the solution.
      - Execution time of the method.
      - A graph showing the allocation of patients to the hospitals.
    - The program also identifies the best solution found among all methods, considering the lowest total cost and the lowest average distance.

**Notes:**

- Execution time may vary depending on the size of the instance and the chosen method.
- The brute force method can take a considerable amount of time for large instances.
- HTML files and graphs are generated in the same folder where the program is located.

-------------------------------------------------------------------------------
""",
    }
}

# Variável global para o idioma
idioma = "pt"

# Função para escolher o idioma
def escolher_idioma():
    global idioma
    print("Escolha o idioma / Choose the language / Seleccione el idioma:")
    print("1 - Português")
    print("2 - Español")
    print("3 - English")
    while True:
        try:
            escolha = int(input("> "))
            if escolha == 1:
                idioma = "pt"
                break
            elif escolha == 2:
                idioma = "es"
                break
            elif escolha == 3:
                idioma = "en"
                break
            else:
                print("Opção inválida / Invalid option / Opción inválida.")
        except ValueError:
            print("Entrada inválida. Digite um número. / Invalid input. Enter a number. / Entrada inválida. Ingrese un número.")
    print(IDIOMAS[idioma]['introducao'])
    
# Função para calcular a distância de Haversine
def haversine(lon1, lat1, lon2, lat2):
    """
    Calcula a distância em quilômetros entre dois pontos a partir de suas coordenadas
    de latitude e longitude.
    """
    # Converter graus decimais para radianos
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Diferença das longitudes e latitudes
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Fórmula de Haversine
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Raio da Terra em quilômetros
    return c * r

# Função para escrever um DataFrame em um arquivo CSV
def escrever_csv(df, nome_arquivo):
    try:
        df.to_csv(nome_arquivo, sep=';', index=False, encoding="utf-8")
        print(f"Arquivo {nome_arquivo} criado com sucesso!")
    except Exception as e:
        print(f"Erro ao escrever o arquivo {nome_arquivo}: {e}")

# Função para ler dados de um arquivo CSV e validar o cabeçalho
def ler_csv(nome_arquivo, cabecalho_esperado):
    cabecalho_esperado_list = [col.strip() for col in cabecalho_esperado.split(";")]
    try:
        # Lê apenas as colunas especificadas no cabeçalho esperado, usando ',' como separador decimal
        df = pd.read_csv(nome_arquivo, sep=";", encoding="utf-8", usecols=cabecalho_esperado_list, decimal=',')

        # Converte as colunas numéricas para float
        for col in df.columns:
            if col in ['Coordenada x', 'Coordenada y', 'Custo', 'Demanda', 'Capacidade']:
                try:
                    # Tenta converter usando ',' como separador decimal
                    df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
                except ValueError:
                    print(f"Aviso: Não foi possível converter a coluna '{col}' para float no arquivo '{nome_arquivo}'. Verifique os dados.")

        # Corrigir cabeçalho se necessário
        cabecalho_atual = [col.strip() for col in df.columns]
        if cabecalho_atual != cabecalho_esperado_list:
            print(IDIOMAS[idioma]["erro_cabecalho_invalido"] + f" {nome_arquivo}. Corrigindo...")

            # Tenta encontrar a melhor correspondência para as colunas
            mapeamento_colunas = {}
            for col_esperada in cabecalho_esperado_list:
                for col_atual in cabecalho_atual:
                    if col_esperada.lower() == col_atual.lower():
                        mapeamento_colunas[col_atual] = col_esperada
                        break

            # Se não conseguir mapear todas as colunas
            if len(mapeamento_colunas) != len(cabecalho_esperado_list):
                print(f"ERRO CRÍTICO: Problema na correção do cabeçalho do arquivo {nome_arquivo}.")
                print(f"Cabeçalho atual: {cabecalho_atual}")
                print(f"Cabeçalho esperado: {cabecalho_esperado_list}")
                raise ValueError(f"Não foi possível corrigir o cabeçalho do arquivo {nome_arquivo}. Verifique o arquivo manualmente.")

            # Renomeia as colunas com base no mapeamento
            df = df.rename(columns=mapeamento_colunas)

            # Força a correção das colunas que não foram mapeadas
            for col in cabecalho_esperado_list:
                if col not in df.columns:
                    df[col] = ""  # Adiciona a coluna faltante com valores vazios

            # Reordena as colunas para coincidir com o cabeçalho esperado
            df = df[cabecalho_esperado_list]
            escrever_csv(df, nome_arquivo)

        return df

    except FileNotFoundError:
        print(IDIOMAS[idioma]["arquivo_nao_encontrado"].format(nome_arquivo=nome_arquivo))
        print(f"Criando o arquivo {nome_arquivo} com cabeçalho padrão.")
        df = pd.DataFrame(columns=cabecalho_esperado_list)
        escrever_csv(df, nome_arquivo)
        return df

    except pd.errors.ParserError as e:
        print(f"Erro de formatação no arquivo CSV {nome_arquivo}: {e}")
        print("Verifique se o arquivo está formatado corretamente e se o separador é ';'.")
        return None

    except ValueError as e:
        # Modificação para lidar com o caso em que as colunas esperadas não estão presentes
        print(f"Erro de validação no arquivo CSV {nome_arquivo}: {e}")
        print(f"As colunas esperadas {cabecalho_esperado_list} não foram encontradas no arquivo.")
        print("Criando um novo arquivo com as colunas necessárias e as colunas encontradas.")
        
        # Tenta ler o arquivo para capturar as colunas existentes
        try:
            df_temp = pd.read_csv(nome_arquivo, sep=";", encoding="utf-8")
            existing_cols = df_temp.columns.tolist()
        except Exception as ex:
            print(f"Erro ao tentar ler o arquivo para obter colunas existentes: {ex}")
            existing_cols = []

        # Combina as colunas esperadas com as colunas existentes (removendo duplicatas)
        all_cols = cabecalho_esperado_list + [col for col in existing_cols if col not in cabecalho_esperado_list]

        # Cria um DataFrame vazio com todas as colunas
        df = pd.DataFrame(columns=all_cols)

        # Escreve o DataFrame vazio no arquivo
        escrever_csv(df, nome_arquivo)
        return df
        
    except Exception as e:
        print(f"Erro inesperado ao ler o arquivo {nome_arquivo}: {e}")
        return None
       
# --- Funções para Entrada de Dados ---
def inserir_dados_pacientes(pacientes_df):
    print(IDIOMAS[idioma]["inserir_pacientes"])
    novos_pacientes = []
    while True:
        nome = input(IDIOMAS[idioma]["nome_paciente"] + " (ou '" + IDIOMAS[idioma]["fim"] + "' para sair): ")
        if nome.lower() == IDIOMAS[idioma]["fim"]:
            break
        try:
            coord_x_str = input(IDIOMAS[idioma]["coord_x"]).replace(',', '.')
            coord_y_str = input(IDIOMAS[idioma]["coord_y"]).replace(',', '.')

            coord_x = float(coord_x_str)
            coord_y = float(coord_y_str)

            demanda = int(input(IDIOMAS[idioma]["demanda"]))
            novos_pacientes.append({'Nome do Paciente': nome, 'Coordenada x': coord_x, 'Coordenada y': coord_y, 'Demanda': demanda})
        except ValueError:
            print("Entrada inválida. Por favor, insira números válidos.")
            continue

    # Cria um novo DataFrame com os dados inseridos
    if novos_pacientes:
        pacientes_df = pd.DataFrame(novos_pacientes)

        # Reordena as colunas para coincidir com o cabeçalho esperado e remove colunas extras
        colunas_pacientes = [s.strip() for s in IDIOMAS[idioma]['cabecalho_pacientes'].split(';')]
        pacientes_df = pacientes_df.reindex(columns=colunas_pacientes)

    return pacientes_df

def inserir_dados_hospitais(hospitais_df):
    print(IDIOMAS[idioma]["inserir_hospitais"])
    novos_hospitais = []
    while True:
        local = input(IDIOMAS[idioma]["local_hospital"] + " (ou '" + IDIOMAS[idioma]["fim"] + "' para sair): ")
        if local.lower() == IDIOMAS[idioma]["fim"]:
            break
        try:
            coord_x_str = input(IDIOMAS[idioma]["coord_x"]).replace(',', '.')
            coord_y_str = input(IDIOMAS[idioma]["coord_y"]).replace(',', '.')
            custo_str = input(IDIOMAS[idioma]["custo"]).replace(',', '.')

            coord_x = float(coord_x_str)
            coord_y = float(coord_y_str)
            capacidade = int(input(IDIOMAS[idioma]["capacidade"]))
            custo = float(custo_str)
            novos_hospitais.append({'Local do Hospital': local, 'Coordenada x': coord_x, 'Coordenada y': coord_y, 'Capacidade': capacidade, 'Custo': custo})
        except ValueError:
            print("Entrada inválida. Por favor, insira números válidos.")
            continue

    # Cria um novo DataFrame com os dados inseridos
    if novos_hospitais:
        hospitais_df = pd.DataFrame(novos_hospitais)

        # Reordena as colunas para coincidir com o cabeçalho esperado e remove colunas extras
        colunas_hospitais = [s.strip() for s in IDIOMAS[idioma]['cabecalho_hospitais'].split(';')]
        hospitais_df = hospitais_df.reindex(columns=colunas_hospitais)

    return hospitais_df

def entrada_dados():
    global idioma
    pacientes_df = None
    hospitais_df = None
    p = None
    while True:
        try:
            opcao = int(input(IDIOMAS[idioma]["opcao_importar"]))
            if opcao == 1:
                # Solicita o nome dos arquivos
                nome_arquivo_pacientes = input(f"Digite o nome do arquivo CSV de pacientes (ou 'c' para cancelar a importação): ")
                if nome_arquivo_pacientes.lower() == 'c':
                    continue  # Volta para o início do loop de entrada de dados

                nome_arquivo_hospitais = input(f"Digite o nome do arquivo CSV de hospitais (ou 'c' para cancelar a importação): ")
                if nome_arquivo_hospitais.lower() == 'c':
                    continue  # Volta para o início do loop de entrada de dados

                # Verifica se os arquivos existem
                pacientes_df = ler_csv(nome_arquivo_pacientes, IDIOMAS[idioma]["cabecalho_pacientes"])
                hospitais_df = ler_csv(nome_arquivo_hospitais, IDIOMAS[idioma]["cabecalho_hospitais"])

                # Se um dos arquivos não for encontrado, cria os arquivos vazios e encerra
                if pacientes_df is None or hospitais_df is None:
                    print("Um ou ambos os arquivos não foram encontrados. Criando arquivos vazios com cabeçalhos padrão.")
                    cabecalho_pacientes_list = [s.strip() for s in IDIOMAS[idioma]["cabecalho_pacientes"].split(";")]
                    cabecalho_hospitais_list = [s.strip() for s in IDIOMAS[idioma]["cabecalho_hospitais"].split(";")]
                    pd.DataFrame(columns=cabecalho_pacientes_list).to_csv(IDIOMAS[idioma]["pacientes"], sep=";", index=False, encoding="utf-8")
                    pd.DataFrame(columns=cabecalho_hospitais_list).to_csv(IDIOMAS[idioma]["hospitais"], sep=";", index=False, encoding="utf-8")
                    print(f"Arquivos {IDIOMAS[idioma]['pacientes']} e {IDIOMAS[idioma]['hospitais']} criados.")
                    print("Por favor, preencha os arquivos com os dados e execute o programa novamente.")
                    return None, None, None  # Encerra o programa

                # Se os arquivos existirem, mas estiverem vazios, solicitar inserção manual
                if pacientes_df.empty:
                    print(f"O arquivo {nome_arquivo_pacientes} está vazio. Insira os dados dos pacientes.")
                    pacientes_df = inserir_dados_pacientes(pacientes_df)
                    escrever_csv(pacientes_df, nome_arquivo_pacientes)

                if hospitais_df.empty:
                    print(f"O arquivo {nome_arquivo_hospitais} está vazio. Insira os dados dos hospitais.")
                    hospitais_df = inserir_dados_hospitais(hospitais_df)
                    escrever_csv(hospitais_df, nome_arquivo_hospitais)

                break

            elif opcao == 2:
                # Cria os DataFrames vazios com os cabeçalhos corretos no idioma selecionado
                pacientes_df = pd.DataFrame(columns=IDIOMAS[idioma]["cabecalho_pacientes"].split(";"))
                hospitais_df = pd.DataFrame(columns=IDIOMAS[idioma]["cabecalho_hospitais"].split(";"))

                pacientes_df = inserir_dados_pacientes(pacientes_df)
                hospitais_df = inserir_dados_hospitais(hospitais_df)
                escrever_csv(pacientes_df, IDIOMAS[idioma]["pacientes"])
                escrever_csv(hospitais_df, IDIOMAS[idioma]["hospitais"])
                print(f"Arquivos {IDIOMAS[idioma]['pacientes']} e {IDIOMAS[idioma]['hospitais']} criados com sucesso!")
                break
            else:
                print("Opção inválida.")
        except ValueError:
            print("Entrada inválida. Digite um número.")

    # Validação do número de pacientes
    while True:
        if pacientes_df is not None and not pacientes_df.empty:
            break
        else:
            print(IDIOMAS[idioma]["erro_pacientes_menor_zero"])
            pacientes_df = pd.DataFrame(columns=IDIOMAS[idioma]["cabecalho_pacientes"].split(";"))
            pacientes_df = inserir_dados_pacientes(pacientes_df)
            escrever_csv(pacientes_df, IDIOMAS[idioma]["pacientes"])

    # Tratamento para o número de p-medianas
    while True:
        try:
            p = int(input(IDIOMAS[idioma]["num_p_medianas"]))
            if hospitais_df is not None and not hospitais_df.empty:
                if p > len(hospitais_df):
                    print(IDIOMAS[idioma]["erro_num_p_maior_hospitais"])
                elif p < 1:
                    print(IDIOMAS[idioma]["erro_num_p_menor_um"])
                else:
                    break
            else:
                print(IDIOMAS[idioma]["erro_hospitais_nao_inicializado"])
                # Sai do loop de entrada de dados se não houver hospitais
                return None, None, None
        except ValueError:
            print("Entrada inválida. Digite um número.")
    print(IDIOMAS[idioma]["fim_entrada_dados"])
    return pacientes_df, hospitais_df, p

# --- Funções para as Abordagens de Resolução ---

# Função para calcular o custo total e a distância média de uma solução (manter como está)
def calcular_custo_distancia(pacientes_df, hospitais_df, centros, alocacao):
    custo_total = sum(hospitais_df.loc[c, 'Custo'] for c in centros)
    distancia_total = 0
    num_pacientes_atendidos = 0
    for paciente, centro in alocacao.items():
        if centro is not None:
            distancia_total += haversine(
                pacientes_df.loc[paciente, 'Coordenada x'],
                pacientes_df.loc[paciente, 'Coordenada y'],
                hospitais_df.loc[centro, 'Coordenada x'],
                hospitais_df.loc[centro, 'Coordenada y']
            )
            num_pacientes_atendidos += 1

    # Correção aqui: Calcula a média corretamente
    distancia_media = distancia_total / num_pacientes_atendidos if num_pacientes_atendidos > 0 else 0
    return custo_total, distancia_media

# 1. Força Bruta
def forca_bruta(pacientes_df, hospitais_df, p, idioma):
    try:
        # VERIFICAÇÃO DOS CABEÇALHOS
        cabecalho_pacientes_esperado = [s.strip() for s in IDIOMAS[idioma]['cabecalho_pacientes'].split(';')]
        cabecalho_hospitais_esperado = [s.strip() for s in IDIOMAS[idioma]['cabecalho_hospitais'].split(';')]

        if list(pacientes_df.columns) != cabecalho_pacientes_esperado:
            raise ValueError(IDIOMAS[idioma]["erro_colunas_faltantes"].format(nome_dataframe="pacientes", lista_colunas=pacientes_df.columns, cabecalho_esperado=cabecalho_pacientes_esperado))
        if list(hospitais_df.columns) != cabecalho_hospitais_esperado:
            raise ValueError(IDIOMAS[idioma]["erro_colunas_faltantes"].format(nome_dataframe="hospitais", lista_colunas=hospitais_df.columns, cabecalho_esperado=cabecalho_hospitais_esperado))

        melhor_solucao = None
        melhor_custo = float('inf')
        melhor_distancia = float('inf')
        melhor_alocacao = {}
        inicio = time.time()

        for combinacao in itertools.combinations(hospitais_df.index, p):
            alocacao = {}
            demanda_atendida = {c: 0 for c in combinacao}
            
            # Lista para armazenar pacientes não alocados
            clientes_nao_alocados = []

            for i, paciente in pacientes_df.iterrows():
                melhor_centro = None
                menor_distancia = float('inf')
                for j in combinacao:
                    distancia = haversine(paciente['Coordenada x'], paciente['Coordenada y'],
                                          hospitais_df.loc[j, 'Coordenada x'], hospitais_df.loc[j, 'Coordenada y'])
                    if distancia < menor_distancia and demanda_atendida[j] + paciente['Demanda'] <= hospitais_df.loc[j, 'Capacidade']:
                        menor_distancia = distancia
                        melhor_centro = j

                if melhor_centro is not None:
                    alocacao[i] = melhor_centro
                    demanda_atendida[melhor_centro] += paciente['Demanda']
                else:
                    clientes_nao_alocados.append(i) # Adiciona paciente à lista de não alocados.

            # Tratamento para clientes não alocados
            for cliente_nao_alocado in clientes_nao_alocados:
                melhor_centro_alternativo = None
                menor_distancia_alternativa = float('inf')

                for j in combinacao:
                    distancia = haversine(pacientes_df.loc[cliente_nao_alocado, 'Coordenada x'],
                                          pacientes_df.loc[cliente_nao_alocado, 'Coordenada y'],
                                          hospitais_df.loc[j, 'Coordenada x'], hospitais_df.loc[j, 'Coordenada y'])

                    if distancia < menor_distancia_alternativa:
                        menor_distancia_alternativa = distancia
                        melhor_centro_alternativo = j

                if melhor_centro_alternativo is not None:
                    alocacao[cliente_nao_alocado] = melhor_centro_alternativo
                    demanda_atendida[melhor_centro_alternativo] += pacientes_df.loc[cliente_nao_alocado, 'Demanda']
                else:
                    alocacao[cliente_nao_alocado] = None

            custo, distancia = calcular_custo_distancia(pacientes_df, hospitais_df, combinacao, alocacao)
            if custo < melhor_custo or (custo == melhor_custo and distancia < melhor_distancia):
                melhor_custo = custo
                melhor_distancia = distancia
                melhor_solucao = combinacao
                melhor_alocacao = alocacao

            if time.time() - inicio > 300:
                raise TimeoutError(IDIOMAS[idioma]["Tempo limite excedido para o método de força bruta."])

        return melhor_solucao, melhor_alocacao, melhor_custo, melhor_distancia, clientes_nao_alocados #Retorna a lista de pacientes não alocados

    except ValueError as e:
        print(f"Erro na Força Bruta: {e}")
        return [], {}, float('inf'), float('inf'), []
    except TimeoutError as e:
        print(e)
        return [], {}, float('inf'), float('inf'), []

# 2. Guloso
def guloso(pacientes_df, hospitais_df, p, idioma):
    try:
        cabecalho_pacientes_esperado = [s.strip() for s in IDIOMAS[idioma]['cabecalho_pacientes'].split(';')]
        cabecalho_hospitais_esperado = [s.strip() for s in IDIOMAS[idioma]['cabecalho_hospitais'].split(';')]

        if list(pacientes_df.columns) != cabecalho_pacientes_esperado:
            raise ValueError("Erro nos cabeçalhos dos pacientes.")
        if list(hospitais_df.columns) != cabecalho_hospitais_esperado:
            raise ValueError("Erro nos cabeçalhos dos hospitais.")

        centros = []
        alocacao = {}
        demanda_atendida = {i: 0 for i in hospitais_df.index}
        pacientes_nao_atendidos = []

        # Seleção gulosa de hospitais
        while len(centros) < p:
            melhor_centro = None
            maior_beneficio = -float('inf')

            for hospital in hospitais_df.index:
                if hospital not in centros:
                    beneficio = sum(
                        1 / (haversine(p['Coordenada x'], p['Coordenada y'],
                                       hospitais_df.loc[hospital, 'Coordenada x'],
                                       hospitais_df.loc[hospital, 'Coordenada y']) + 1e-5)
                        for _, p in pacientes_df.iterrows()
                        if demanda_atendida[hospital] + p['Demanda'] <= hospitais_df.loc[hospital, 'Capacidade']
                    )
                    if beneficio > maior_beneficio:
                        maior_beneficio = beneficio
                        melhor_centro = hospital

            if melhor_centro is not None:
                centros.append(melhor_centro)

        # Alocar pacientes aos hospitais
        for i, paciente in pacientes_df.iterrows():
            melhor_centro = None
            menor_distancia = float('inf')

            for centro in centros:
                if demanda_atendida[centro] + paciente['Demanda'] <= hospitais_df.loc[centro, 'Capacidade']:
                    distancia = haversine(paciente['Coordenada x'], paciente['Coordenada y'],
                                          hospitais_df.loc[centro, 'Coordenada x'], hospitais_df.loc[centro, 'Coordenada y'])
                    if distancia < menor_distancia:
                        menor_distancia = distancia
                        melhor_centro = centro

            if melhor_centro is not None:
                alocacao[i] = melhor_centro
                demanda_atendida[melhor_centro] += paciente['Demanda']
            else:
                pacientes_nao_atendidos.append(i)

        # Rechecagem para alocar pacientes restantes
        for paciente_id in pacientes_nao_atendidos[:]:
            paciente = pacientes_df.loc[paciente_id]
            for centro in centros:
                if demanda_atendida[centro] + paciente['Demanda'] <= hospitais_df.loc[centro, 'Capacidade']:
                    alocacao[paciente_id] = centro
                    demanda_atendida[centro] += paciente['Demanda']
                    pacientes_nao_atendidos.remove(paciente_id)
                    break

        custo, distancia = calcular_custo_distancia(pacientes_df, hospitais_df, centros, alocacao)
        return centros, alocacao, custo, distancia, pacientes_nao_atendidos

    except Exception as e:
        print(f"Erro no método Guloso: {e}")
        return [], {}, float('inf'), float('inf'), []

# 3. Programação Dinâmica
def programacao_dinamica(pacientes_df, hospitais_df, p, idioma):
    try:
        cabecalho_pacientes_esperado = [s.strip() for s in IDIOMAS[idioma]['cabecalho_pacientes'].split(';')]
        cabecalho_hospitais_esperado = [s.strip() for s in IDIOMAS[idioma]['cabecalho_hospitais'].split(';')]

        if list(pacientes_df.columns) != cabecalho_pacientes_esperado:
            raise ValueError("Erro nos cabeçalhos dos pacientes.")
        if list(hospitais_df.columns) != cabecalho_hospitais_esperado:
            raise ValueError("Erro nos cabeçalhos dos hospitais.")

        n = len(hospitais_df)
        distancias = {
            (i, j): haversine(
                pacientes_df.loc[i, 'Coordenada x'], pacientes_df.loc[i, 'Coordenada y'],
                hospitais_df.loc[j, 'Coordenada x'], hospitais_df.loc[j, 'Coordenada y']
            )
            for i in pacientes_df.index for j in hospitais_df.index
        }

        dp = {}
        for mascara in range(1 << n):
            dp[mascara] = (float('inf'), None)

        dp[0] = (0, [])

        for mascara in range(1 << n):
            custo_atual, centros = dp[mascara]
            for nova_instalacao in hospitais_df.index:
                if (mascara & (1 << nova_instalacao)) == 0:
                    nova_mascara = mascara | (1 << nova_instalacao)
                    novo_custo = custo_atual + hospitais_df.loc[nova_instalacao, 'Custo']
                    nova_alocacao = centros + [nova_instalacao]
                    if novo_custo < dp[nova_mascara][0]:
                        dp[nova_mascara] = (novo_custo, nova_alocacao)

        melhor_custo = float('inf')
        melhor_solucao = None

        for mascara, (custo, centros) in dp.items():
            if bin(mascara).count('1') == p and custo < melhor_custo:
                melhor_custo = custo
                melhor_solucao = centros

        alocacao = {}
        demanda_atendida = {c: 0 for c in melhor_solucao}
        pacientes_nao_atendidos = []

        for i, paciente in pacientes_df.iterrows():
            melhor_centro = None
            menor_distancia = float('inf')

            for centro in melhor_solucao:
                if demanda_atendida[centro] + paciente['Demanda'] <= hospitais_df.loc[centro, 'Capacidade']:
                    distancia = distancias[(i, centro)]
                    if distancia < menor_distancia:
                        menor_distancia = distancia
                        melhor_centro = centro

            if melhor_centro is not None:
                alocacao[i] = melhor_centro
                demanda_atendida[melhor_centro] += paciente['Demanda']
            else:
                pacientes_nao_atendidos.append(i)

        # Rechecagem para alocar pacientes restantes
        for paciente_id in pacientes_nao_atendidos[:]:
            paciente = pacientes_df.loc[paciente_id]
            for centro in melhor_solucao:
                if demanda_atendida[centro] + paciente['Demanda'] <= hospitais_df.loc[centro, 'Capacidade']:
                    alocacao[paciente_id] = centro
                    demanda_atendida[centro] += paciente['Demanda']
                    pacientes_nao_atendidos.remove(paciente_id)
                    break

        distancia_total = sum(
            distancias[(i, alocacao[i])] for i in alocacao if alocacao[i] is not None
        )
        distancia_media = distancia_total / len(pacientes_df) if len(pacientes_df) > 0 else 0

        return melhor_solucao, alocacao, melhor_custo, distancia_media, pacientes_nao_atendidos

    except Exception as e:
        print(f"Erro na Programação Dinâmica: {e}")
        return [], {}, float('inf'), float('inf'), []

# 4. Divisão e Conquista
def divisao_conquista(pacientes_df, hospitais_df, p, idioma):
    try:
        cabecalho_pacientes_esperado = [s.strip() for s in IDIOMAS[idioma]['cabecalho_pacientes'].split(';')]
        cabecalho_hospitais_esperado = [s.strip() for s in IDIOMAS[idioma]['cabecalho_hospitais'].split(';')]

        if list(pacientes_df.columns) != cabecalho_pacientes_esperado:
            raise ValueError("Erro nos cabeçalhos dos pacientes.")
        if list(hospitais_df.columns) != cabecalho_hospitais_esperado:
            raise ValueError("Erro nos cabeçalhos dos hospitais.")

        def dividir_geograficamente(pacientes, hospitais):
            centro_x = pacientes['Coordenada x'].median()
            pacientes_esq = pacientes[pacientes['Coordenada x'] <= centro_x]
            pacientes_dir = pacientes[pacientes['Coordenada x'] > centro_x]
            hospitais_esq = hospitais[hospitais['Coordenada x'] <= centro_x]
            hospitais_dir = hospitais[hospitais['Coordenada x'] > centro_x]
            return (pacientes_esq, hospitais_esq), (pacientes_dir, hospitais_dir)

        def resolver_recursivo(pacientes, hospitais, p):
            if len(hospitais) == 1 or p == 1:
                return guloso(pacientes, hospitais, p, idioma)

            (pacientes_esq, hospitais_esq), (pacientes_dir, hospitais_dir) = dividir_geograficamente(pacientes, hospitais)
            p_esq = max(1, min(p - 1, round(p * len(hospitais_esq) / len(hospitais))))
            p_dir = p - p_esq

            centros_esq, alocacao_esq, custo_esq, distancia_esq, nao_atendidos_esq = resolver_recursivo(pacientes_esq, hospitais_esq, p_esq)
            centros_dir, alocacao_dir, custo_dir, distancia_dir, nao_atendidos_dir = resolver_recursivo(pacientes_dir, hospitais_dir, p_dir)

            centros = centros_esq + centros_dir
            alocacao = {**alocacao_esq, **alocacao_dir}
            pacientes_nao_atendidos = nao_atendidos_esq + nao_atendidos_dir

            demanda_atendida = {c: 0 for c in centros}
            for paciente_id, centro in alocacao.items():
                if centro is not None:
                    demanda_atendida[centro] += pacientes.loc[paciente_id, 'Demanda']

            for paciente_id in pacientes_nao_atendidos[:]:
                paciente = pacientes.loc[paciente_id]
                for centro in centros:
                    if demanda_atendida[centro] + paciente['Demanda'] <= hospitais.loc[centro, 'Capacidade']:
                        alocacao[paciente_id] = centro
                        demanda_atendida[centro] += paciente['Demanda']
                        pacientes_nao_atendidos.remove(paciente_id)
                        break

            custo = sum(hospitais.loc[c, 'Custo'] for c in centros)
            distancia_total = sum(
                haversine(pacientes.loc[i, 'Coordenada x'], pacientes.loc[i, 'Coordenada y'],
                          hospitais.loc[alocacao[i], 'Coordenada x'], hospitais.loc[alocacao[i], 'Coordenada y'])
                for i in alocacao if alocacao[i] is not None
            )
            distancia_media = distancia_total / len(pacientes) if len(pacientes) > 0 else 0

            return centros, alocacao, custo, distancia_media, pacientes_nao_atendidos

        return resolver_recursivo(pacientes_df, hospitais_df, p)

    except Exception as e:
        print(f"Erro na Divisão e Conquista: {e}")
        return [], {}, float('inf'), float('inf'), []

# --- Função para Gerar a Saída em HTML ---
def gerar_html(pacientes_df, hospitais_df, centros, alocacao, custo, distancia, metodo, tempo_execucao, melhor_solucao=None, idioma=idioma, nao_atendidos=None):
    data_geracao = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Criação do gráfico
    plt.figure(figsize=(10, 8))

    # Definir cores para os centros
    cmap = plt.get_cmap('jet')
    cores = cmap(np.linspace(0, 1, max(len(centros), 1)))

    # Plota os pacientes
    pacientes_atendidos = []
    for i, paciente in pacientes_df.iterrows():
        if i in alocacao and alocacao[i] is not None:
            pacientes_atendidos.append(i)
            centro = alocacao[i]
            cor = cores[centros.index(centro)] if centro in centros else 'gray'
            plt.plot([paciente[IDIOMAS[idioma]['coord_x'].strip()], hospitais_df.loc[centro, IDIOMAS[idioma]['coord_x'].strip()]],
                     [paciente[IDIOMAS[idioma]['coord_y'].strip()], hospitais_df.loc[centro, IDIOMAS[idioma]['coord_y'].strip()]],
                     color=cor, alpha=0.5, linewidth=0.5)
            plt.scatter(paciente[IDIOMAS[idioma]['coord_x'].strip()], paciente[IDIOMAS[idioma]['coord_y'].strip()], color=cor, edgecolors='k', s=20, marker='o')
        else:
            # Plota os pacientes não alocados com a cor cinza
            plt.scatter(paciente[IDIOMAS[idioma]['coord_x'].strip()], paciente[IDIOMAS[idioma]['coord_y'].strip()], color='gray', edgecolors='k', s=20, marker='o')

    # Plota os centros logísticos
    for i, centro in enumerate(hospitais_df.index):
        if centro in centros:
            # Plota os centros logísticos utilizados com o marcador '*' e cor específica
            cor_centro = cores[centros.index(centro)]
            plt.scatter(hospitais_df.loc[centro, IDIOMAS[idioma]['coord_x'].strip()], hospitais_df.loc[centro, IDIOMAS[idioma]['coord_y'].strip()], s=100, color=cor_centro, marker='*', edgecolors='k', linewidths=1)
        else:
            # Plota os centros logísticos não utilizados com o marcador 'X' e cor cinza
            plt.scatter(hospitais_df.loc[centro, IDIOMAS[idioma]['coord_x'].strip()], hospitais_df.loc[centro, IDIOMAS[idioma]['coord_y'].strip()], s=100, color='gray', marker='X', edgecolors='k', linewidths=1)

    # Adiciona a legenda para os centros
    legend_labels = [plt.Line2D([0], [0], marker='o', color='w', label=IDIOMAS[idioma]['grafico_pacientes'],
                                 markerfacecolor='gray', markersize=10, markeredgecolor='k'),
                     plt.Line2D([0], [0], marker='*', color='w', label=IDIOMAS[idioma]['grafico_hospitais'],
                                 markerfacecolor='red', markersize=15, markeredgecolor='k'),
                     plt.Line2D([0], [0], marker='X', color='w', label=f'{IDIOMAS[idioma]["hospitais_a"]} - {IDIOMAS[idioma]["nao_utilizado"]}',
                                 markerfacecolor='gray', markersize=10, markeredgecolor='k')]

    # Adiciona a legenda para cada centro, diferenciando os utilizados dos não utilizados
    for i, centro in enumerate(hospitais_df.index):
        nome_centro = hospitais_df.loc[centro, IDIOMAS[idioma]['local_hospital'].strip()]
        if centro in centros:
            cor_centro = cores[centros.index(centro)]
            legend_labels.append(plt.Line2D([0], [0], marker='*', color='w', label=f'{nome_centro} ({IDIOMAS[idioma]["hospital"]})',
                                           markerfacecolor=cor_centro, markersize=15, markeredgecolor='k'))
        else:
            legend_labels.append(plt.Line2D([0], [0], marker='X', color='w', label=f'{nome_centro} ({IDIOMAS[idioma]["nao_utilizado"]})',
                                           markerfacecolor='gray', markersize=10, markeredgecolor='k'))

    plt.legend(handles=legend_labels, bbox_to_anchor=(1.05,1), loc='best', title=IDIOMAS[idioma]['grafico_legenda'])
    plt.title(f'{IDIOMAS[idioma]["grafico_titulo"]} - {metodo}')
    plt.xlabel(IDIOMAS[idioma]['coord_x'])
    plt.ylabel(IDIOMAS[idioma]['coord_y'])

    # Salva o gráfico como imagem
    nome_arquivo_grafico = f"grafico_{metodo.replace(' ', '_')}_{idioma}_{datetime.datetime.now().strftime('%d%m%Y_%H%M')}.png"
    plt.savefig(nome_arquivo_grafico, bbox_inches='tight')
    plt.close()

    # Adicionar um título informativo sobre o método usado
    titulo_metodo = f"<h2>{IDIOMAS[idioma]['metodo']}: {metodo}</h2>"

    # Adicionar informações detalhadas sobre a melhor solução encontrada
    info_melhor_solucao = ""
    if melhor_solucao:
        info_melhor_solucao = f"""
        <h3>{IDIOMAS[idioma]['melhor_solucao_titulo']}:</h3>
        <p><b>{IDIOMAS[idioma]['melhor_solucao_metodo']}:</b> {melhor_solucao['metodo']}</p>
        <p><b>{IDIOMAS[idioma]['melhor_solucao_distancia']}:</b> {melhor_solucao['distancia']:.2f}{IDIOMAS[idioma]['km']}</p>
        <p><b>{IDIOMAS[idioma]['melhor_solucao_custo']}:</b> {melhor_solucao['custo']:.2f}{IDIOMAS[idioma]['dinheiro']}</p>
        """

    # Informações sobre a solução atual
    info_solucao_atual = f"""
    <h3>{IDIOMAS[idioma]['solucao_z']} ({metodo}):</h3>
    <p><b>{IDIOMAS[idioma]['tempo']}:</b> {tempo_execucao:.2f} segundos</p>
    <p><b>{IDIOMAS[idioma]['distancia_media']}:</b> {distancia:.2f}{IDIOMAS[idioma]['km']}</p>
    <p><b>{IDIOMAS[idioma]['custo_total']}:</b> {custo:.2f}{IDIOMAS[idioma]['dinheiro']}</p>
    """

    # Tabela com os centros logísticos
    tabela_centros = f"""
    <h3>{IDIOMAS[idioma]['hospitais_a']}:</h3>
    <table border="1">
        <tr>
            <th>{IDIOMAS[idioma]['local_hospital']}</th>
            <th>{IDIOMAS[idioma]['coord_x']}</th>
            <th>{IDIOMAS[idioma]['coord_y']}</th>
            <th>{IDIOMAS[idioma]['custo']}</th>
        </tr>
    """
    for centro in centros:
        tabela_centros += f"""
        <tr>
            <td>{hospitais_df.loc[centro, IDIOMAS[idioma]['local_hospital'].strip()]}</td>
            <td>{hospitais_df.loc[centro, IDIOMAS[idioma]['coord_x'].strip()]}</td>
            <td>{hospitais_df.loc[centro, IDIOMAS[idioma]['coord_y'].strip()]}</td>
            <td>{hospitais_df.loc[centro, IDIOMAS[idioma]['custo'].strip()]:.2f}{IDIOMAS[idioma]['dinheiro']}</td>
        </tr>
        """
    tabela_centros += "</table>"

    # Tabela com os pacientes atendidos por cada centro
    tabela_pacientes = f"""
    <h3>{IDIOMAS[idioma]['pacientes_atendidos']}</h3>
    <table border="1">
        <tr>
            <th>{IDIOMAS[idioma]['hospital']}</th>
            <th>{IDIOMAS[idioma]['pacientes']}</th>
        </tr>
    """
    for centro in centros:
        pacientes_atendidos_nomes = [pacientes_df.loc[paciente, IDIOMAS[idioma]['nome_paciente'].strip()] for paciente, c in alocacao.items() if c == centro]
        tabela_pacientes += f"""
        <tr>
            <td>{hospitais_df.loc[centro, IDIOMAS[idioma]['local_hospital'].strip()]}</td>
            <td>{', '.join(pacientes_atendidos_nomes)}</td>
        </tr>
        """
    tabela_pacientes += "</table>"
    
    # Tabela com os pacientes não atendidos
    tabela_pacientes_nao_atendidos = f"""
    <h3>{IDIOMAS[idioma]['pacientes']} não atendidos</h3>
    <table border="1">
        <tr>
            <th>{IDIOMAS[idioma]['nome_paciente']}</th>
            <th>{IDIOMAS[idioma]['coord_x']}</th>
            <th>{IDIOMAS[idioma]['coord_y']}</th>
            <th>{IDIOMAS[idioma]['demanda']}</th>
        </tr>
    """
    if nao_atendidos is None:
        nao_atendidos = []

    for paciente in nao_atendidos:
        tabela_pacientes_nao_atendidos += f"""
        <tr>
            <td>{pacientes_df.loc[paciente, IDIOMAS[idioma]['nome_paciente'].strip()]}</td>
            <td>{pacientes_df.loc[paciente, IDIOMAS[idioma]['coord_x'].strip()]}</td>
            <td>{pacientes_df.loc[paciente, IDIOMAS[idioma]['coord_y'].strip()]}</td>
            <td>{pacientes_df.loc[paciente, IDIOMAS[idioma]['demanda'].strip()]}</td>
        </tr>
        """
    tabela_pacientes_nao_atendidos += "</table>"

    # Resumo da solução
    total_capacidade = sum(hospitais_df.loc[c, 'Capacidade'] for c in centros)
    
    resumo_solucao = f"""
    <h3>Resumo da Solução</h3>
    <table border="1">
        <tr>
            <th>Total de Pacientes Atendidos</th>
            <td>{len(pacientes_atendidos)}</td>
        </tr>
        <tr>
            <th>Total de Pacientes Não Atendidos</th>
            <td>{len(nao_atendidos)}</td>
        </tr>
        <tr>
            <th>Capacidade Total dos Hospitais Utilizados</th>
            <td>{total_capacidade}</td>
        </tr>
    </table>
    """

    # Explicação da melhor solução, se disponível
    explicacao_melhor_solucao = ""
    if melhor_solucao:
        if melhor_solucao['metodo'] == "Força Bruta":
            explicacao_melhor_solucao = f"<p>{IDIOMAS[idioma]['melhor_solucao_explicacao_fb']}</p>"
        elif melhor_solucao['metodo'] == "Guloso":
            explicacao_melhor_solucao = f"<p>{IDIOMAS[idioma]['melhor_solucao_explicacao_g']}</p>"
        elif melhor_solucao['metodo'] == "Programação Dinâmica":
            explicacao_melhor_solucao = f"<p>{IDIOMAS[idioma]['melhor_solucao_explicacao_pd']}</p>"
        elif melhor_solucao['metodo'] == "Divisão e Conquista":
            explicacao_melhor_solucao = f"<p>{IDIOMAS[idioma]['melhor_solucao_explicacao_dc']}</p>"
    
    # Inclui o gráfico no HTML
    grafico_html = f'<img src="{nome_arquivo_grafico}" alt="{IDIOMAS[idioma]["grafico_titulo"]}">'

    # Montando o HTML final
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{IDIOMAS[idioma]['resultado_otimizacao']}</title>
        <meta charset="UTF-8">
    </head>
    <body>
        <h1>{IDIOMAS[idioma]['relatorio_otimizacao']} - {IDIOMAS[idioma]['problema_p_medianas']}</h1>
        <p>{IDIOMAS[idioma]['data_geracao']}{data_geracao}</p>
        {titulo_metodo}
        {info_melhor_solucao}
        {explicacao_melhor_solucao}
        {info_solucao_atual}
        {tabela_centros}
        {tabela_pacientes}
        {tabela_pacientes_nao_atendidos}
        {resumo_solucao}
        {grafico_html}
    </body>
    </html>
    """

    return html

# --- Função Principal ---
def main():
    global idioma
    escolher_idioma()

    while True:
        pacientes_df, hospitais_df, p = entrada_dados()

        # Verifica se entrada_dados retornou valores válidos
        if pacientes_df is None or hospitais_df is None or p is None:
            print("Erro durante a entrada de dados. Reiniciando o programa.")
            return  # Encerra o programa se houver erro na entrada de dados

        if pacientes_df.empty or hospitais_df.empty:
            print("Erro: Os dados de pacientes ou hospitais estão vazios. Reiniciando o programa.")
            continue

        # VERIFICAÇÃO E CORREÇÃO DE NOMES DE COLUNAS APÓS entrada_dados
        pacientes_df.columns = [col.replace(":", "").strip() for col in pacientes_df.columns]
        hospitais_df.columns = [col.replace(":", "").strip() for col in hospitais_df.columns]

        # Verifica se os dataframes não estão vazios e se possuem as colunas necessárias
        if pacientes_df.empty:
            print("Erro: DataFrame de pacientes vazio. Reiniciando o programa.")
            continue
        if hospitais_df.empty:
            print("Erro: DataFrame de hospitais vazio. Reiniciando o programa.")
            continue

        # Verifica se os dataframes têm as colunas necessárias
        pacientes_cols = [s.strip() for s in IDIOMAS[idioma]['cabecalho_pacientes'].split(';')]
        if not all(col in pacientes_df.columns for col in pacientes_cols):
            print(f"Erro: DataFrame de pacientes não possui as colunas necessárias. Reiniciando o programa.")
            continue
        hospitais_cols = [s.strip() for s in IDIOMAS[idioma]['cabecalho_hospitais'].split(';')]
        if not all(col in hospitais_df.columns for col in hospitais_cols):
            print(f"Erro: DataFrame de hospitais não possui as colunas necessárias. Reiniciando o programa.")
            continue

        resultados = {}

        # Força Bruta
        print(IDIOMAS[idioma]["calculando_solucao"] + " Força Bruta...")
        inicio = time.time()
        try:
            centros_fb, alocacao_fb, custo_fb, distancia_fb, nao_atendidos_fb = forca_bruta(pacientes_df, hospitais_df, p, idioma)
            tempo_fb = time.time() - inicio
            resultados["Força Bruta"] = {"centros": centros_fb, "alocacao": alocacao_fb, "custo": custo_fb, "distancia": distancia_fb, "tempo": tempo_fb, "nao_atendidos": nao_atendidos_fb}
        except TimeoutError:
            print(f"Força Bruta: {IDIOMAS[idioma]['solucao_otima_nao_encontrada']}")
            resultados["Força Bruta"] = {"centros": [], "alocacao": {}, "custo": float('inf'), "distancia": float('inf'), "tempo": time.time() - inicio, "nao_atendidos": []}

        # Guloso
        print(IDIOMAS[idioma]["calculando_solucao"] + " Guloso...")
        inicio = time.time()
        try:
            centros_g, alocacao_g, custo_g, distancia_g, nao_atendidos_g = guloso(pacientes_df, hospitais_df, p, idioma)
            tempo_g = time.time() - inicio
            resultados["Guloso"] = {"centros": centros_g, "alocacao": alocacao_g, "custo": custo_g, "distancia": distancia_g, "tempo": tempo_g, "nao_atendidos": nao_atendidos_g}
        except TimeoutError:
            print(f"Guloso: {IDIOMAS[idioma]['solucao_otima_nao_encontrada']}")
            resultados["Guloso"] = {"centros": [], "alocacao": {}, "custo": float('inf'), "distancia": float('inf'), "tempo": time.time() - inicio, "nao_atendidos": []}

        # Programação Dinâmica
        print(IDIOMAS[idioma]["calculando_solucao"] + " Programação Dinâmica...")
        inicio = time.time()
        try:
            centros_pd, alocacao_pd, custo_pd, distancia_pd, nao_atendidos_pd = programacao_dinamica(pacientes_df, hospitais_df, p, idioma)
            tempo_pd = time.time() - inicio
            resultados["Programação Dinâmica"] = {"centros": centros_pd, "alocacao": alocacao_pd, "custo": custo_pd, "distancia": distancia_pd, "tempo": tempo_pd, "nao_atendidos": nao_atendidos_pd}
        except TimeoutError:
            print(f"Programação Dinâmica: {IDIOMAS[idioma]['solucao_otima_nao_encontrada']}")
            resultados["Programação Dinâmica"] = {"centros": [], "alocacao": {}, "custo": float('inf'), "distancia": float('inf'), "tempo": time.time() - inicio, "nao_atendidos": []}

        # Divisão e Conquista
        print(IDIOMAS[idioma]["calculando_solucao"] + " Divisão e Conquista...")
        inicio = time.time()
        try:
            centros_dc, alocacao_dc, custo_dc, distancia_dc, nao_atendidos_dc = divisao_conquista(pacientes_df, hospitais_df, p, idioma)
            tempo_dc = time.time() - inicio
            resultados["Divisão e Conquista"] = {"centros": centros_dc, "alocacao": alocacao_dc, "custo": custo_dc, "distancia": distancia_dc, "tempo": tempo_dc, "nao_atendidos": nao_atendidos_dc}
        except TimeoutError:
            print(f"Divisão e Conquista: {IDIOMAS[idioma]['solucao_otima_nao_encontrada']}")
            resultados["Divisão e Conquista"] = {"centros": [], "alocacao": {}, "custo": float('inf'), "distancia": float('inf'), "tempo": time.time() - inicio, "nao_atendidos": []}

        # Encontrar a melhor solução
        melhor_solucao = None
        for metodo, resultado in resultados.items():
            if melhor_solucao is None:
                melhor_solucao = {"metodo": metodo, "custo": resultado["custo"], "distancia": resultado["distancia"]}
            elif (resultado["custo"] < melhor_solucao["custo"]) or \
                    (resultado["custo"] == melhor_solucao["custo"] and resultado["distancia"] < melhor_solucao["distancia"]):
                melhor_solucao = {"metodo": metodo, "custo": resultado["custo"], "distancia": resultado["distancia"]}

        for metodo, resultado in resultados.items():
            # Gerar HTML para cada método
            nome_arquivo_html = f"resultado_{metodo.replace(' ', '_')}_{idioma}_{datetime.datetime.now().strftime('%d%m%Y_%H%M')}.html"
            with open(nome_arquivo_html, "w", encoding="utf-8") as f:
                f.write(gerar_html(pacientes_df, hospitais_df, resultado["centros"], resultado["alocacao"], resultado["custo"], resultado["distancia"], metodo, resultado["tempo"], melhor_solucao, idioma, resultado["nao_atendidos"]))

        # Exibe a melhor solução e por que ela foi considerada a melhor
        if melhor_solucao is not None:
            print(f"\n{IDIOMAS[idioma]['melhor_solucao_titulo']}: {melhor_solucao['metodo']}")
            if melhor_solucao["metodo"] == "Força Bruta":
                print(IDIOMAS[idioma]['melhor_solucao_explicacao_fb'])
            elif melhor_solucao["metodo"] == "Guloso":
                print(IDIOMAS[idioma]['melhor_solucao_explicacao_g'])
            elif melhor_solucao["metodo"] == "Programação Dinâmica":
                print(IDIOMAS[idioma]['melhor_solucao_explicacao_pd'])
            elif melhor_solucao["metodo"] == "Divisão e Conquista":
                print(IDIOMAS[idioma]['melhor_solucao_explicacao_dc'])
        else:
            print("Nenhuma solução encontrada.")

        # Pergunta ao usuário se deseja refazer a operação
        while True:
            opcao = input(IDIOMAS[idioma]["refazer_operacao"])
            if opcao == "1":
                break  # Sai do loop interno para refazer a operação
            elif opcao == "2":
                print(IDIOMAS[idioma]["encerrando_programa"])
                return  # Encerra a execução da função main atual
            else:
                print("Opção inválida.")

if __name__ == "__main__":
    main()
    