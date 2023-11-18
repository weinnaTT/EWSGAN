"""GanAlgorithm."""
from __future__ import absolute_import, division, print_function
import copy
import os
import numpy as np
import random


# To implement easy, wo use int number in the code to represent the
# corresponding operations. These are replaced with one-hot in the paper,
# which can better explain our theory. Each int number is equal to a one-hot number.


class GanAlgorithm():

    def __init__(self, args):
        self.steps = 3
        self.up_nodes = 2
        self.down_nodes = 2
        self.normal_nodes = 5
        self.dis_normal_nodes = 5
        self.archs = {}
        self.dis_archs = {}
        self.base_dis_arch = np.array(
            [[3, 0, 3, 0, 2, 0, 0], [3, 0, 3, 0, 1, -1, -1], [3, 0, 3, 0, 1, -1, -1]])
        self.Normal_G = []
        self.Up_G = []
        self.Normal_G_fixed = []
        self.Up_G_fixed = []
        for i in range(self.steps * self.normal_nodes):
            self.Normal_G.append([0, 1, 2, 3, 4, 5, 6])
        for i in range(self.steps * self.up_nodes):
            self.Up_G.append([0, 1, 2])
        for i in range(self.steps * self.normal_nodes):
            self.Normal_G_fixed.append([0, 1, 2, 3, 4, 5, 6])  # ([0,1,2,3,4,5,6])
        for i in range(self.steps * self.up_nodes):
            self.Up_G_fixed.append([0, 1, 2])

    def Get_Operation(self, Candidate_Operation, mode, num, remove=True):
        '''
        Candidate_Operation: operation pool
        mode: Operation type
        return: the operation
        remove：remove from the pool
        '''
        if Candidate_Operation == [] and mode == 'up':
            Candidate_Operation += self.Up_G_fixed[num]
        elif Candidate_Operation == [] and mode == 'normal':
            Candidate_Operation += self.Normal_G_fixed[num]
        choice = random.choice(Candidate_Operation)
        if remove:
            Candidate_Operation.remove(choice)
        return choice

    def uniform_sample(self):
        genotype = np.zeros(
            (self.steps, self.up_nodes + self.normal_nodes), dtype=int)
        for i in range(self.steps):
            for j in range(self.up_nodes):
                genotype[i][j] = random.randint(0, 2)
            while (genotype[i][2] == 0 and genotype[i][3] == 0):
                for k in range(2):
                    genotype[i][k + 2] = random.randint(0, 6)
            while (genotype[i][4] == 0 and genotype[i][5] == 0 and genotype[i][6] == 0):
                for k in range(2, self.normal_nodes):
                    genotype[i][k + 2] = random.randint(0, 6)
        return genotype

    def sample_fair(self, remove=True):
        genotype = np.zeros(
            (self.steps, self.up_nodes + self.normal_nodes), dtype=int)  # 3*7
        for i in range(self.steps):
            for j in range(self.up_nodes):
                genotype[i][j] = self.Get_Operation(self.Up_G[2 * i + j], 'up', 2 * i + j, remove)

            for k in range(2):
                genotype[i][k + 2] = self.Get_Operation(self.Normal_G[5 * i + k], 'normal', 5 * i + k, remove)
            while (genotype[i][2] == 0 and genotype[i][3] == 0):
                for k in range(2):
                    genotype[i][k + 2] = self.Get_Operation(self.Normal_G[5 * i + k], 'normal', 5 * i + k, False)

            for k in range(2, self.normal_nodes):
                genotype[i][k + 2] = self.Get_Operation(self.Normal_G[5 * i + k], 'normal', 5 * i + k, remove)
            while (genotype[i][4] == 0 and genotype[i][5] == 0 and genotype[i][6] == 0):
                for k in range(2, self.normal_nodes):
                    genotype[i][k + 2] = self.Get_Operation(self.Normal_G[5 * i + k], 'normal', 5 * i + k, False)

        return genotype

    def sample_zero(self, remove=True):
        genotype = np.zeros(
            (self.steps, self.up_nodes + self.normal_nodes), dtype=int)
        for i in range(self.steps):
            for j in range(self.up_nodes):
                genotype[i][j] = self.Get_Operation(self.Up_G[2 * i + j], 'up', 2 * i + j, remove)
        return genotype

    def encode(self, genotype):
        lists = [0 for i in range(self.steps)]
        for i in range(len(lists)):
            lists[i] = str(genotype[i])
        return tuple(lists)

    def search(self, remove=True):
        new_genotype = self.sample_fair(remove)
        # new_genotype = self.uniform_sample()
        return new_genotype

    def sample_D(self):
        genotype = np.zeros((self.steps, self.down_nodes +
                             self.normal_nodes), dtype=int)
        for i in range(self.steps):
            genotype[i][0] = random.randint(1, 6)
            while (genotype[i][1] == 0 and genotype[i][2] == 0):
                for k in range(2):
                    genotype[i][k + 2] = random.randint(0, 6)
            while (genotype[i][3] == 0 and genotype[i][4] == 0):
                for k in range(2, 4):
                    genotype[i][k + 1] = random.randint(0, 6)
            genotype[i][self.dis_normal_nodes] = random.randint(-1, 5)
            if genotype[i][self.dis_normal_nodes] == -1:
                genotype[i][self.dis_normal_nodes + 1] = -1
            else:
                genotype[i][self.dis_normal_nodes + 1] = random.randint(0, 5)
        return genotype

    def judge_repeat(self, new_genotype):
        t = self.encode(new_genotype)
        return t in self.archs

    def judge_repeat_dis(self, new_genotype):
        t = self.encode(new_genotype)
        return t in self.dis_archs

    def search_mutate_dis(self):
        t = self.encode(self.base_dis_arch)
        self.dis_archs[t] = self.base_dis_arch
        while (t in self.dis_archs):
            new_genotype = self.mutation_gen(self.base_dis_arch)
            t = self.encode(new_genotype)
        self.dis_archs[t] = new_genotype
        return new_genotype

    def mutation_dis(self, alphas_a, ratio=0.5):
        """Mutation for an individual"""
        new_alphas = alphas_a.copy()
        layer = random.randint(0, self.steps - 1)
        index = random.randint(0, self.down_nodes + self.dis_normal_nodes - 1)
        if index == 0:
            new_alphas[layer][index] = random.randint(1, 6)
            while (new_alphas[layer][index] == alphas_a[layer][index]):
                new_alphas[layer][index] = random.randint(1, 6)
        elif index >= 1 and index < 3:
            new_alphas[layer][index] = random.randint(0, 6)
            while (new_alphas[layer][1] == 0 and new_alphas[layer][2] == 0) or (
                    new_alphas[layer][index] == alphas_a[layer][index]):
                new_alphas[layer][index] = random.randint(0, 6)
        elif index >= 3 and index < 5:
            new_alphas[layer][index] = random.randint(0, 6)
            while (new_alphas[layer][3] == 0 and new_alphas[layer][4] == 0) or (
                    new_alphas[layer][index] == alphas_a[layer][index]):
                new_alphas[layer][index] = random.randint(0, 6)
        if index == 5:
            new_alphas[layer][index] = random.randint(-1, 5)
            while (new_alphas[layer][index] == alphas_a[layer][index]):
                new_alphas[layer][index] = random.randint(-1, 5)
        if index == 6:
            new_alphas[layer][index] = random.randint(0, 5)
            while (new_alphas[layer][index] == alphas_a[layer][index]):
                new_alphas[layer][index] = random.randint(0, 5)
        return new_alphas

    def mutation_gen(self, alphas_a, ratio=0.5):
        """Mutation for an individual"""
        new_alphas = alphas_a.copy()
        layer = random.randint(0, self.steps - 1)
        index = random.randint(0, self.down_nodes + self.normal_nodes - 1)
        if index < 2:
            new_alphas[layer][index] = random.randint(0, 2)
            while (new_alphas[layer][index] == alphas_a[layer][index]):
                new_alphas[layer][index] = random.randint(0, 2)
        elif index >= 2 and index < 4:
            new_alphas[layer][index] = random.randint(0, 6)
            while (new_alphas[layer][2] == 0 and new_alphas[layer][3] == 0) or (
                    new_alphas[layer][index] == alphas_a[layer][index]):
                new_alphas[layer][index] = random.randint(0, 6)
        elif index >= 4:
            new_alphas[layer][index] = random.randint(0, 6)
            while (new_alphas[layer][3] == 0 and new_alphas[layer][4] == 0) or (
                    new_alphas[layer][index] == alphas_a[layer][index]):
                new_alphas[layer][index] = random.randint(0, 6)
        return new_alphas
