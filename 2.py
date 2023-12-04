import random

# 게임 규칙 설정
N = 6  # 염색체 길이 (숫자야구 숫자의 자릿수)
MAX_GENERATIONS = 100  # 최대 세대 수
POPULATION_SIZE = 10  # 염색체 개수

def generate_chromosome():
    """랜덤한 N자리 숫자로 염색체 생성"""
    return [random.randint(0, 9) for _ in range(N)]

def calculate_fitness(solution, guess):
    """적합도 계산: 스트라이크를 5점, 볼을 1점으로 환산"""
    strikes = sum(s == g for s, g in zip(solution, guess))
    balls = sum(s in guess and s != g for s, g in zip(solution, guess))
    return strikes * 5 + balls

def select_parents(population, fitness_scores):
    """룰렛 방식으로 부모 선택"""
    total_fitness = sum(fitness_scores)
    selection_probabilities = [fitness / total_fitness for fitness in fitness_scores]
    parents = random.choices(population, weights=selection_probabilities, k=2)
    return parents

def crossover(parent1, parent2):
    """교차 연산을 통해 자손 생성"""
    crossover_point = random.randint(1, N - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

def mutate(child):
    """돌연변이 연산을 통해 자손의 일부를 변형"""
    mutation_point = random.randint(0, N - 1)
    child[mutation_point] = random.randint(0, 9)
    return child

# 초기 염색체 생성
population = [generate_chromosome() for _ in range(POPULATION_SIZE)]

for generation in range(MAX_GENERATIONS):
    # 적합도 계산 및 출력
    fitness_scores = [calculate_fitness(solution=[1, 2, 3, 4, 5, 6], guess=chromosome) for chromosome in population]
    best_chromosome = population[fitness_scores.index(max(fitness_scores))]
    print(f"세대 {generation + 1}, 최적 염색체: {best_chromosome}, 적합도: {max(fitness_scores)}")

    # 다음 세대를 위해 새로운 염색체 생성
    new_population = []

    for _ in range(POPULATION_SIZE // 2):
        # 부모 선택
        parent1, parent2 = select_parents(population, fitness_scores)

        # 교차 연산을 통해 자손 생성
        child1 = crossover(parent1, parent2)
        child2 = crossover(parent2, parent1)

        # 돌연변이 연산
        child1 = mutate(child1)
        child2 = mutate(child2)

        new_population.extend([child1, child2])

    # 새로 생성된 자손으로 염색체 갱신
    population = new_population

# 최종 결과 출력
best_solution = population[fitness_scores.index(max(fitness_scores))]
print(f"최적 해: {best_solution}")
