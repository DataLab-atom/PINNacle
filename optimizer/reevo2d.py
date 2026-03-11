from typing import Optional, Callable
import logging
import subprocess
import numpy as np
import os
from omegaconf import DictConfig
import random
from utils.utils import *
from utils.llm_client.base import BaseClient
import copy
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed

class ReEv2d:
    def __init__(
        self,
        cfg: DictConfig,
        root_dir: str,
        generator_llm: BaseClient,
        reflector_llm: Optional[BaseClient] = None,
        get_funcs: Optional[Callable] = None,
        evaluate_funcs: Optional[Callable] = None,
    ) -> None:
        self.cfg = cfg
        self.generator_llm = generator_llm
        self.reflector_llm = reflector_llm or generator_llm
        self.root_dir = root_dir
        self.get_funcs = get_funcs
        self.evaluate_funcs = evaluate_funcs
        
        self.mutation_rate = cfg.mutation_rate
        self.iteration = 0
        self.function_evals = 0
        self.elitist = None
        self.long_term_reflection_str = ""
        self.best_obj_overall = None
        self.best_code_overall = None
        self.best_code_path_overall = None
        self.best_code_func_index_overall = None
        self.reevo_func_index = self.cfg.reevo_func_index

        self.init_prompt()
        self.func_elitist = {i:[] for i in self.func_names}
        self.func_individual = {i:[] for i in self.func_names}
        self.func_best_obj_overall = {i:1e10 for i in self.func_names}
        self.func_best_code_overall = {}
        self.func_best_code_path_overall = {}
        self.init_population()
        


    def init_prompt(self) -> None:
        self.problem = self.cfg.problem.problem_name
        self.problem_desc = self.cfg.problem.description
        self.problem_size = self.cfg.problem.problem_size
        # self.func_name = self.cfg.problem.func_name
        self.obj_type = self.cfg.problem.obj_type
        self.problem_type = self.cfg.problem.problem_type

        logging.info("Problem: " + self.problem)
        logging.info("Problem description: " + self.problem_desc)
        # logging.info("Function name: " + self.func_name)

        self.prompt_dir = f"{self.root_dir}/prompts"
        self.output_dir = f"{self.root_dir}/problems/{self.problem}/"
        
        # Loading all text prompts
        # Problem-specific prompt components
        prompt_path_suffix = "_black_box" if self.problem_type == "black_box" else ""
        problem_prompt_path = f'{self.prompt_dir}/{self.problem}{prompt_path_suffix}'


        # ----------------------------------------------------------------------------------------------------------------------
        if self.get_funcs is not None:
            # API 模式：调用外部函数获取可优化函数列表
            self.init_generated_funcs = self.get_funcs()
            logging.info("Loaded functions from get_funcs() API.")
        else:
            # 文件模式：从 .npy 文件加载
            self.init_generated_funcs = np.load(f'{self.output_dir}/init_generated_funcs.npy', allow_pickle=True)
        self.func_names = [i['func_name'] for i in self.init_generated_funcs]

        self.func_signatures = []
        self.seed_funcs = []
        self.func_descs = []
        self.func_docs = []
        self.external_knowledges = []
        for i in self.init_generated_funcs:
            self.seed_funcs.append(i['func_source'])
            self.func_signatures.append(i['func_source'].split('\n')[0])
            self.func_descs.append(i['func_description'])
            self.func_docs.append(i['doc'])
            if 'external_knowledge' in i.keys():
                self.external_knowledges.append(i['external_knowledge'])
            else:
                self.external_knowledges.append("")
        
        logging.info("Functions name: [" + ','.join(self.func_names) + ']')
        if self.cfg.algorithm != 'reevo2d':
            iter_func_names = [self.func_names[i] for i in self.reevo_func_index]
            logging.info("Functions name in Iter: [" + ','.join(iter_func_names) + ']')
        # ----------------------------------------------------------------------------------------------------------------------
        

        # self.seed_func = file_to_string(f'{problem_prompt_path}/seed_func.txt')
        # self.func_signature = file_to_string(f'{problem_prompt_path}/func_signature.txt')
        # self.func_desc = file_to_string(f'{problem_prompt_path}/func_desc.txt')
        # if os.path.exists(f'{problem_prompt_path}/external_knowledge.txt'):
        #     self.external_knowledge = file_to_string(f'{problem_prompt_path}/external_knowledge.txt')
        #     self.long_term_reflection_str = self.external_knowledge
        # else:
        #     self.external_knowledge = ""
        
        
        # Common prompts
        self.system_generator_prompt = file_to_string(f'{self.prompt_dir}/common/system_generator.txt')
        self.system_reflector_prompt = file_to_string(f'{self.prompt_dir}/common/system_reflector.txt')
        self.user_reflector_st_prompt = file_to_string(f'{self.prompt_dir}/common/user_reflector_st.txt') if self.problem_type != "black_box" else file_to_string(f'{self.prompt_dir}/common/user_reflector_st_black_box.txt') # shrot-term reflection
        self.user_reflector_lt_prompt = file_to_string(f'{self.prompt_dir}/common/user_reflector_lt.txt') # long-term reflection
        self.crossover_prompt = file_to_string(f'{self.prompt_dir}/common/crossover.txt')
        self.mutataion_prompt = file_to_string(f'{self.prompt_dir}/common/mutation.txt')
        
        # ----------------------------------------------------------------------------------------------------------------------
        self.user_generator_prompts = []
        self.seed_prompts = []
        for func_name,func_desc,seed_func,doc in zip(self.func_names,self.func_descs,self.seed_funcs,self.func_docs):
            self.user_generator_prompts.append(file_to_string(f'{self.prompt_dir}/common/user_generator.txt').format(
                func_name=func_name, 
                problem_desc=self.problem_desc,
                func_desc=func_desc,
                doc= doc
            ))
            self.seed_prompts.append(file_to_string(f'{self.prompt_dir}/common/seed.txt').format(
                seed_func=seed_func,
                func_name=func_name,
            ))

        # ----------------------------------------------------------------------------------------------------------------------

        # self.user_generator_prompt = file_to_string(f'{self.prompt_dir}/common/user_generator.txt').format(
        #     func_name=self.func_name, 
        #     problem_desc=self.problem_desc,
        #     func_desc=self.func_desc,
        #     )
        # self.seed_prompt = file_to_string(f'{self.prompt_dir}/common/seed.txt').format(
        #     seed_func=self.seed_func,
        #     func_name=self.func_name,
        # )

        # Flag to print prompts
        self.print_crossover_prompt = True # Print crossover prompt for the first iteration
        self.print_mutate_prompt = True # Print mutate prompt for the first iteration
        self.print_short_term_reflection_prompt = True # Print short-term reflection prompt for the first iteration
        self.print_long_term_reflection_prompt = True # Print long-term reflection prompt for the first iteration
        

    def init_population(self) -> None:
        # Evaluate the seed function, and set it as Elite
        logging.info("Evaluating seed function...")
        code = extract_code_from_generators(self.seed_funcs)
        logging.info("Seed function code: \n" + code)
        seed_ind = {
            "stdout_filepath": f"problem_iter{self.iteration}_stdout0.txt",
            "code_path": f"problem_iter{self.iteration}_code0.py",
            "code": code,
            "funcs": self.seed_funcs,
            "func_index":-1,
            "response_id": 0,
        }
        self.seed_ind = seed_ind
        self.population = self.evaluate_population([seed_ind])*len(self.func_names)
        for i,_ in enumerate(self.func_names):
            self.population[i]['func_index'] = i

        # If seed function is invalid, stop
        if not self.seed_ind["exec_success"]:
            raise RuntimeError(f"Seed function is invalid. Please check the stdout file in {os.getcwd()}.")

        self.update_iter()
        
        # Generate responses
        system = self.system_generator_prompt
        population = []
        for index,(user_generator_prompt,seed_prompt) in enumerate(zip(self.user_generator_prompts,self.seed_prompts)):
            if self.cfg.algorithm != 'reevo2d':
                if not index in self.reevo_func_index:
                    continue
            user = user_generator_prompt + "\n" + seed_prompt + "\n" + self.long_term_reflection_str
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            logging.info("Initial Population Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)

            responses = self.generator_llm.multi_chat_completion([messages], self.cfg.init_pop_size, temperature = self.generator_llm.temperature + 0.3) # Increase the temperature for diverse initial population
            population += [self.response_to_individual_2d(response, response_id,None,self.seed_funcs,index) for response_id, response in enumerate(responses)]

        # Run code and evaluate population
        population = self.evaluate_population(population)

        # Update iteration
        self.population = population
        self.update_iter()

    
    # def response_to_individual(self, response: str, response_id: int, file_name: str=None) -> dict:
    #     """
    #     Convert response to individual
    #     """
    #     # Write response to file
    #     file_name = f"problem_iter{self.iteration}_response{response_id}.txt" if file_name is None else file_name + ".txt"
    #     with open(file_name, 'w') as file:
    #         file.writelines(response + '\n')

    #     code = extract_code_from_generator(response)

    #     # Extract code and description from response
    #     std_out_filepath = f"problem_iter{self.iteration}_stdout{response_id}.txt" if file_name is None else file_name + "_stdout.txt"
        
    #     individual = {
    #         "stdout_filepath": std_out_filepath,
    #         "code_path": f"problem_iter{self.iteration}_code{response_id}.py",
    #         "code": code,
    #         "response_id": response_id,
    #     }
    #     return individual
    
    def response_to_individual_2d(self, response: str, response_id: int, file_name: str=None, funcs:list = [],func_index: int = 0) -> dict:
        """
        Convert response to individual
        """
        # Write response to file
        file_name = f"problem_iter{self.iteration}_response{response_id}.txt" if file_name is None else file_name + ".txt"
        try:
            with open(file_name, 'w') as file:
                file.writelines(response + '\n')
        except:
            try:
                with open(file_name, 'w',encoding='utf-8') as file:
                    file.writelines(response + '\n')
            except:
                with open(file_name, 'w',encoding='utf-8') as file:
                    file.writelines("" + '\n')
            

        # Extract code and description from response
        std_out_filepath = f"problem_iter{self.iteration}_stdout{response_id}.txt" if file_name is None else file_name + "_stdout.txt"
        
        temp_funcs = copy.deepcopy(funcs)
        temp_funcs[func_index] = extract_code_from_generator_onece(response).replace('_v2','').replace('_v1','')
        code = extract_code_from_generators(temp_funcs)
        
        individual = {
            "stdout_filepath": std_out_filepath,
            "code_path": f"problem_iter{self.iteration}_funcIndex{func_index}_code{response_id}.txt",
            "code": code,
            "funcs":temp_funcs,
            "func_index":func_index,
            "response_id": response_id,
        }
        return individual

    def mark_invalid_individual(self, individual: dict, traceback_msg: str) -> dict:
        """
        Mark an individual as invalid.
        """
        individual["exec_success"] = False
        individual["obj"] = float("inf")
        individual["traceback_msg"] = traceback_msg
        return individual

    # -----------------------------------------------------------------------------------------------------------------------
    # API 评估模式：使用外部 evaluate_funcs 函数替代子进程 eval.py
    # -----------------------------------------------------------------------------------------------------------------------

    def _build_func_list_for_eval(self, individual: dict) -> list:
        """
        根据 individual 构建传给 evaluate_funcs 的函数列表。
        对被修改的函数（func_index 对应的那个）更新 func_source 并标记 is_modified=True，
        其余函数保持原样并标记 is_modified=False。
        func_index=-1 时表示 seed（全部未修改）。
        """
        func_list = []
        for i, func_dict in enumerate(self.init_generated_funcs):
            entry = dict(func_dict)
            if individual["func_index"] != -1 and i == individual["func_index"]:
                entry["func_source"] = individual["funcs"][i]
                entry["is_modified"] = True
            else:
                entry["is_modified"] = False
            func_list.append(entry)
        return func_list

    def _aggregate_scores(self, scores: list) -> float:
        """
        将 evaluate_funcs 返回的多指标列表聚合为单一 obj 值（始终以最小化为目标）。

        每个 score 元素格式（evaluate_funcs 的返回约定）：
        {
            "name":      str,          # 指标名称
            "value":     float,        # 指标数值
            "direction": "min"/"max",  # 优化方向
            "weight":    float,        # 可选，默认 1.0
        }
        对 direction=="max" 的指标取负值，从而统一转为最小化问题。
        """
        total = 0.0
        for s in scores:
            value = float(s["value"])
            direction = s.get("direction", "min")
            weight = float(s.get("weight", 1.0))
            total += weight * value if direction == "min" else -weight * value
        return total

    def _evaluate_with_api(self, individual: dict) -> dict:
        """
        使用 evaluate_funcs API 评估单个 individual。
        """
        try:
            func_list = self._build_func_list_for_eval(individual)
            scores = self.evaluate_funcs(func_list)
            obj = self._aggregate_scores(scores)
            individual["obj"] = obj
            individual["exec_success"] = True
            individual["scores"] = scores
            logging.info(f"Iteration {self.iteration}: API eval obj={obj}, scores={scores}")
        except Exception as e:
            logging.info(f"API eval error: {e}")
            individual = self.mark_invalid_individual(individual, str(e))
        return individual

    def _evaluate_population_with_api(self, population: list[dict]) -> list[dict]:
        """
        使用 evaluate_funcs API 并发批量评估种群。

        每个 individual 提交给独立线程，evaluate_funcs 在隔离临时目录中运行，
        多个调用之间没有共享文件状态，并发安全。
        """
        # 预处理：标记无效个体，收集需要真正评估的下标
        valid_indices = []
        for response_id, individual in enumerate(population):
            self.function_evals += 1
            if individual.get("code") is None:
                population[response_id] = self.mark_invalid_individual(individual, "Invalid response!")
            else:
                valid_indices.append(response_id)

        if not valid_indices:
            return population

        # 并发评估：线程数取有效个体数，上限由 CPU 核心数自然约束
        with ThreadPoolExecutor(max_workers=len(valid_indices)) as executor:
            future_to_id = {
                executor.submit(self._evaluate_with_api, population[i]): i
                for i in valid_indices
            }
            for future in as_completed(future_to_id):
                response_id = future_to_id[future]
                logging.info(f"Iteration {self.iteration}: API evaluated individual {response_id}")
                try:
                    population[response_id] = future.result()
                except Exception as e:
                    logging.info(f"Iteration {self.iteration}: individual {response_id} eval error: {e}")
                    population[response_id] = self.mark_invalid_individual(
                        population[response_id], str(e)
                    )

        return population

    # -----------------------------------------------------------------------------------------------------------------------

    def evaluate_population(self, population: list[dict]) -> list[float]:
        """
        Evaluate population by running code in parallel and computing objective values.
        如果传入了 evaluate_funcs，则走 API 评估路径；否则走原始子进程路径。
        """
        if self.evaluate_funcs is not None:
            return self._evaluate_population_with_api(population)

        inner_runs = []
        # Run code to evaluate
        for response_id in range(len(population)):
            self.function_evals += 1
            # Skip if response is invalid
            if population[response_id]["code"] is None:
                population[response_id] = self.mark_invalid_individual(population[response_id], "Invalid response!")
                inner_runs.append(None)
                continue
            
            logging.info(f"Iteration {self.iteration}: Running Code {response_id}")
            
            try:
                process = self._run_code(population[response_id], response_id)
                inner_runs.append(process)
            except Exception as e: # If code execution fails
                logging.info(f"Error for response_id {response_id}: {e}")
                population[response_id] = self.mark_invalid_individual(population[response_id], str(e))
                inner_runs.append(None)
        
        # Update population with objective values
        for response_id, inner_run in enumerate(inner_runs):
            if inner_run is None: # If code execution fails, skip
                continue
            try:
                inner_run.communicate(timeout=self.cfg.timeout) # Wait for code execution to finish
            except subprocess.TimeoutExpired as e:
                logging.info(f"Error for response_id {response_id}: {e}")
                population[response_id] = self.mark_invalid_individual(population[response_id], str(e))
                inner_run.kill()
                continue

            individual = population[response_id]
            stdout_filepath = individual["stdout_filepath"]
            with open(stdout_filepath, 'r') as f:  # read the stdout file
                stdout_str = f.read() 
            traceback_msg = filter_traceback(stdout_str)
            
            individual = population[response_id]
            # Store objective value for each individual
            if traceback_msg == '': # If execution has no error
                try:
                    individual["obj"] = float(stdout_str.split('\n')[-2]) if self.obj_type == "min" else -float(stdout_str.split('\n')[-2])
                    individual["exec_success"] = True
                except:
                    population[response_id] = self.mark_invalid_individual(population[response_id], "Invalid std out / objective value!")
            else: # Otherwise, also provide execution traceback error feedback
                population[response_id] = self.mark_invalid_individual(population[response_id], traceback_msg)

            if np.isinf(individual["obj"]) and np.random.rand() < 0.8:
                individual["exec_success"] = False

            if np.isnan(individual["obj"]):
                individual["obj"] = np.inf
                individual["exec_success"] = False
            
            logging.info(f"Iteration {self.iteration}, response_id {response_id}: Objective value: {individual['obj']}")
        return population


    # def _run_code(self, individual: dict, response_id) -> subprocess.Popen:
    #     """
    #     Write code into a file and run eval script.
    #     """
    #     logging.debug(f"Iteration {self.iteration}: Processing Code Run {response_id}")
        
    #     with open(self.output_file, 'w') as file:
    #         file.writelines(individual["code"] + '\n')

    #     # Execute the python file with flags
    #     with open(individual["stdout_filepath"], 'w') as f:
    #         eval_file_path = f'{self.root_dir}/problems/{self.problem}/eval.py' if self.problem_type != "black_box" else f'{self.root_dir}/problems/{self.problem}/eval_black_box.py' 
    #         process = subprocess.Popen(['python', '-u', eval_file_path, f'{self.problem_size}', self.root_dir, "train"],
    #                                     stdout=f, stderr=f)

    #     block_until_running(individual["stdout_filepath"], log_status=True, iter_num=self.iteration, response_id=response_id)
    #     return process
    

    # ------------------------------------------------------------------------------------------------------------------------
    def _run_code(self, individual: dict, response_id) -> subprocess.Popen:
        """
        Write code into a file and run eval script.
        """
        logging.debug(f"Iteration {self.iteration}: Processing Code Run {response_id}")
        output_index = f'_{individual["func_index"]}'
        outfile_path = f'iter_num_{self.iteration}_func_index{output_index}_response_id_{response_id}.py'
        with open(self.output_dir+'generated/'+outfile_path, 'w') as file:
            file.write(individual["code"])

        # Execute the python file with flags
        with open(individual["stdout_filepath"], 'w') as f:
            eval_file_path = f'{self.root_dir}/problems/{self.problem}/eval.py' if self.problem_type != "black_box" else f'{self.root_dir}/problems/{self.problem}/eval_black_box.py' 
            # print(['python', '-u', eval_file_path, f'{self.problem_size}', self.root_dir, "train",outfile_path])
            process = subprocess.Popen(['python', '-u', eval_file_path, f'{self.problem_size}', self.root_dir, "train",outfile_path],
                                        stdout=f, stderr=f)

        block_until_running(individual["stdout_filepath"], log_status=True, iter_num=self.iteration, response_id=response_id)
        return process
    
    # ------------------------------------------------------------------------------------------------------------------------
    
    def update_iter(self,upper = 1) -> None:
        """
        Update after each iteration
        """
        population = [individual for individual in self.population if individual['exec_success']]
        objs = [individual["obj"] for individual in population]

        if len(objs)<1:
            self.iteration += upper
            return 

        best_obj, best_sample_idx = min(objs), np.argmin(np.array(objs))
        
        # update best overall
        if self.best_obj_overall is None or best_obj <= self.best_obj_overall:
            self.best_obj_overall = best_obj
            self.best_code_overall = population[best_sample_idx]["code"]
            self.best_code_path_overall = population[best_sample_idx]["code_path"]
            self.best_code_func_index_overall = population[best_sample_idx]["func_index"]
        
        # update elitist
        if self.elitist is None or best_obj <= self.elitist["obj"]:
            self.elitist = population[best_sample_idx]
            logging.info(f"Iteration {self.iteration}: Elitist: {self.elitist['obj']}")
        
        logging.info(f"Iteration {self.iteration} finished...")
        logging.info(f"Best obj: {self.best_obj_overall},Best obj func index: {self.best_code_func_index_overall}, Best Code Path: {self.best_code_path_overall}")
        logging.info(f"Function Evals: {self.function_evals}")

        # -----------------------------------------------------------------------------------------------------------------------
        func_individual = [[] for _ in range(len(self.func_names))]
        for individual in population:
            func_individual[individual["func_index"]].append(individual)
        
        for func_index,func_population in enumerate(func_individual):
            if func_population == []:
                continue
            func_name = self.func_names[func_index]
            self.func_individual[func_name] = func_population

            objs = [individual["obj"] for individual in func_population]
            if objs == []:
                continue
            
            best_obj, best_sample_idx = min(objs), np.argmin(np.array(objs))
            
            if best_obj <= self.func_best_obj_overall[func_name]:
                self.func_best_obj_overall[func_name] = best_obj
                self.func_best_code_overall[func_name] = func_population[best_sample_idx]["code"]
                self.func_best_code_path_overall[func_name] = func_population[best_sample_idx]["code_path"]
                self.func_elitist[func_name] = func_population[best_sample_idx]
        
        self.iteration += upper
        
    def rank_select(self, population: list[dict]) -> list[dict]:
        """
        Rank-based selection, select individuals with probability proportional to their rank.
        """
        if self.problem_type == "black_box":
            population = [individual for individual in population if individual["exec_success"] and individual["obj"] < self.seed_ind["obj"]]
        else:
            population = [individual for individual in population if individual["exec_success"]]
        if len(population) < 2:
            return None
        # Sort population by objective value
        population = sorted(population, key=lambda x: x["obj"])
        ranks = [i for i in range(len(population))]
        probs = [1 / (rank + 1 + len(population)) for rank in ranks]
        # Normalize probabilities
        probs = [prob / sum(probs) for prob in probs]
        selected_population = []
        trial = 0
        while len(selected_population) < 2 * self.cfg.pop_size:
            trial += 1
            parents = np.random.choice(population, size=2, replace=False, p=probs)
            if parents[0]["obj"] != parents[1]["obj"]:
                selected_population.extend(parents)
            if trial > 1000:
                return None
        return selected_population
    
    
    def random_select(self, population: list[dict]) -> list[dict]:
        """
        Random selection, select individuals with equal probability.
        """
        selected_population = []
        # Eliminate invalid individuals
        if self.problem_type == "black_box":
            population = [individual for individual in population if individual["exec_success"] and individual["obj"] < self.seed_ind["obj"]]
        else:
            population = [individual for individual in population if individual["exec_success"]]

        if len(population) < 2:
            return None
        trial = 0
        while len(selected_population) < 2 * self.cfg.pop_size:
            trial += 1
            parents = np.random.choice(population, size=2, replace=False)
            # If two parents have the same objective value, consider them as identical; otherwise, add them to the selected population
            if parents[0]["obj"] != parents[1]["obj"]:
                selected_population.extend(parents)
            elif (np.random.rand()<0.2) and (parents[0]["code"] != parents[1]["code"]):
                selected_population.extend(parents)
            if trial > 1000:
                return None
        return selected_population

    def gen_short_term_reflection_prompt(self, ind1: dict, ind2: dict) -> tuple[list[dict], str, str]:
        """
        Short-term reflection before crossovering two individuals.
        """
        #if ind1["obj"] == ind2["obj"]:
        #    print(ind1["code"], ind2["code"])
        #    raise ValueError("Two individuals to crossover have the same objective value!")
        # Determine which individual is better or worse
        if ind1["obj"] <= ind2["obj"]:
            better_ind, worse_ind = ind1, ind2
        elif ind1["obj"] > ind2["obj"]:
            better_ind, worse_ind = ind2, ind1

        func_index = worse_ind['func_index']
        func_name,func_desc = self.func_names[func_index],self.func_descs[func_index]
        
        worse_code = filter_code(worse_ind["funcs"][func_index])
        better_code = filter_code(better_ind["funcs"][func_index])
        
        system = self.system_reflector_prompt
        user = self.user_reflector_st_prompt.format(
            func_name = func_name,
            func_desc = func_desc,
            problem_desc = self.problem_desc,
            worse_code=worse_code,
            better_code=better_code
            )
        message = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        
        # Print reflection prompt for the first iteration
        if self.print_short_term_reflection_prompt:
                logging.info("Short-term Reflection Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
                self.print_short_term_reflection_prompt = False
        return message, worse_code, better_code


    def short_term_reflection(self, population: list[dict]) -> tuple[list[list[dict]], list[str], list[str]]:
        """
        Short-term reflection before crossovering two individuals.
        """
        messages_lst = []
        worse_code_lst = []
        better_code_lst = []
        for i in range(0, len(population), 2):
            # Select two individuals
            parent_1 = population[i]
            parent_2 = population[i+1]
            
            # Short-term reflection
            messages, worse_code, better_code = self.gen_short_term_reflection_prompt(parent_1, parent_2)
            messages_lst.append(messages)
            worse_code_lst.append(worse_code)
            better_code_lst.append(better_code)
        
        # Asynchronously generate responses
        response_lst = self.reflector_llm.multi_chat_completion(messages_lst)
        return response_lst, worse_code_lst, better_code_lst
    
    def long_term_reflection(self, short_term_reflections: list[str]) -> None:
        """
        Long-term reflection before mutation.
        """
        system = self.system_reflector_prompt
        user = self.user_reflector_lt_prompt.format(
            problem_desc = self.problem_desc,
            prior_reflection = self.long_term_reflection_str,
            new_reflection = "\n".join(short_term_reflections),
            )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        
        if self.print_long_term_reflection_prompt:
            logging.info("Long-term Reflection Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
            self.print_long_term_reflection_prompt = False
        
        self.long_term_reflection_str = self.reflector_llm.multi_chat_completion([messages])[0]
        
        # Write reflections to file
        file_name = f"problem_iter{self.iteration}_short_term_reflections.txt"
        with open(file_name, 'w') as file:
            file.writelines("\n".join(short_term_reflections) + '\n')
        
        file_name = f"problem_iter{self.iteration}_long_term_reflection.txt"
        with open(file_name, 'w') as file:
            file.writelines(self.long_term_reflection_str + '\n')


    def crossover(self, short_term_reflection_tuple: tuple[list[list[dict]], list[str], list[str]]) -> list[dict]:
        reflection_content_lst, worse_code_lst, better_code_lst = short_term_reflection_tuple
        messages_lst = []
        for reflection, worse_code, better_code in zip(reflection_content_lst, worse_code_lst, better_code_lst):
            # Crossover
            system = self.system_generator_prompt
            func_signature0 = self.func_signature
            func_signature1 = self.func_signature
            user = self.crossover_prompt.format(
                user_generator = self.user_generator_prompt,
                func_signature0 = func_signature0,
                func_signature1 = func_signature1,
                worse_code = worse_code,
                better_code = better_code,
                reflection = reflection,
                func_name = self.func_name,
            )
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            messages_lst.append(messages)
            
            # Print crossover prompt for the first iteration
            if self.print_crossover_prompt:
                logging.info("Crossover Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
                self.print_crossover_prompt = False
        
        # Asynchronously generate responses
        response_lst = self.generator_llm.multi_chat_completion(messages_lst)
        crossed_population = [self.response_to_individual_2d(response, response_id,None,self.parent_funcs,self.func_index) for response_id, response in enumerate(response_lst)]

        assert len(crossed_population) == self.cfg.pop_size
        return crossed_population


    def mutate(self) -> list[dict]:
        """Elitist-based mutation. We only mutate the best individual to generate n_pop new individuals."""
        system = self.system_generator_prompt
        func_signature1 = self.func_signature
        user = self.mutataion_prompt.format(
            user_generator = self.user_generator_prompt,
            reflection = self.long_term_reflection_str + self.external_knowledge,
            func_signature1 = func_signature1,
            elitist_code = filter_code(self.elitist["code"]),
            func_name = self.func_name,
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if self.print_mutate_prompt:
            logging.info("Mutation Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
            self.print_mutate_prompt = False
        responses = self.generator_llm.multi_chat_completion([messages], int(self.cfg.pop_size * self.mutation_rate))
        population = [self.response_to_individual_2d(response, response_id,None,self.parent_funcs,self.func_index) for response_id, response in enumerate(responses)]
        return population


    def evolve(self):
        while self.function_evals < self.cfg.max_fe:
            # If all individuals are invalid, stop
            if all([not individual["exec_success"] for individual in self.population]):
                raise RuntimeError(f"All individuals are invalid. Please check the stdout files in {os.getcwd()}.")
            
            population = []
            # ------------------------------------------------------------------------------------------------------------------------------------
            ind_map = {k:[i for i in self.func_individual[k] if (i and i['obj'] < 1e23 and i["exec_success"] )] for k in self.func_names}
            ind_keys = [k for k,v in ind_map.items() if len(v) > 5]
            if (self.cfg.algorithm == 'reevo2d') and (len(ind_keys) > 2) :
                responeses,parent_funcs,crossover_index = [],[],[]
                MIN_LENGTH = 20
                for (key_1,key_2) in combinations(ind_keys, 2):
                    ind_1,ind_2 = copy.deepcopy(ind_map[key_1]),copy.deepcopy(ind_map[key_2])
                    random.shuffle(ind_1)
                    random.shuffle(ind_2)
                    min_length = min(len(ind_1),len(ind_2),MIN_LENGTH)
                    key_index_1,key_index_2 = self.func_names.index(key_1),self.func_names.index(key_2)
                    for i in range(min_length):
                        responeses.extend([ind_1[i]['funcs'][key_index_1],ind_2[i]['funcs'][key_index_2]])
                        parent_funcs.extend([ind_2[i]['funcs'],ind_1[i]['funcs']])
                        crossover_index.extend([key_index_1,key_index_2])
                
                for response_id,(response,parent_funcs,index) in enumerate(zip(responeses,parent_funcs,crossover_index)):
                    population.append(self.response_to_individual_2d(response, 100+response_id,None,parent_funcs,index)) 
                self.population = self.evaluate_population(population)

            for key,val in copy.deepcopy(self.func_individual).items():
                func_index = self.func_names.index(key)
                if self.cfg.algorithm != 'reevo2d':
                    if not func_index in self.reevo_func_index:
                        continue
                    self.population = []
                population_to_select = val if (self.elitist is None or self.elitist in val) else [self.elitist] + val # add elitist to population for selection
                selected_population = self.random_select(population_to_select)
                if selected_population is None: 
                    continue
                self.func_index = func_index
                self.parent_funcs = selected_population[0]['funcs']
                self.func_signature = self.func_signatures[func_index]
                self.user_generator_prompt = self.user_generator_prompts[func_index]
                self.external_knowledge = self.external_knowledges[func_index]
                self.func_name = key
                
                # Short-term reflection
                short_term_reflection_tuple = self.short_term_reflection(selected_population) # (response_lst, worse_code_lst, better_code_lst)
                # Crossover
                crossed_population = self.crossover(short_term_reflection_tuple)
                # Evaluate
                self.population.extend(self.evaluate_population(crossed_population))
                # Long-term reflection
                self.long_term_reflection([response for response in short_term_reflection_tuple[0]])
                # Mutate
                mutated_population = self.mutate()
                # Evaluate
                self.population.extend(self.evaluate_population(mutated_population))
            # Update
            self.update_iter()
        return self.best_code_overall, self.best_code_path_overall
