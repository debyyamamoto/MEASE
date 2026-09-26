import random as rd


class PopulationGenerator:
    def gen_interval(self, continuous_attribute):
        if continuous_attribute >= 0:
            lower_interval = continuous_attribute - rd.uniform(0, (continuous_attribute * 0.1))
            upper_interval = continuous_attribute + rd.uniform(0, (continuous_attribute * 0.1))
        else:
            lower_interval = continuous_attribute + rd.uniform(0, (continuous_attribute * 0.1))
            upper_interval = continuous_attribute - rd.uniform(0, (continuous_attribute * 0.1))

        return [lower_interval, upper_interval]

    def gen_ind(self, dataset, row_size):
        individual = [[], []]

        row = rd.randint(0, len(dataset) - 1)
        attributes_quantity = rd.randint(1, 4)

        for i in range(attributes_quantity):
            while True:
                col = rd.randint(0, row_size - 1)
                if col not in individual[0]:
                    break

            if type(dataset[row][col]) == str:
                individual[0].append(col)
                individual[1].append([dataset[row][col]])
            else:
                interval = self.gen_interval(dataset[row][col])
                individual[0].append(col)
                individual[1].append(interval)

        return individual

    def gen_population(self, population_size, dataset):
        population = []
        row_size = len(dataset[0])

        for i in range(population_size):
            individual = self.gen_ind(dataset, row_size)
            population.append(individual)

        return population
