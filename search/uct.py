import math
from collections import OrderedDict
from time import time

import chess
import numpy as np
from search.util import cp

FPU = 0.0
FPU_ROOT = 0.0


class UCTNode(object):
    def __init__(self, board=None, parent=None, move=None, prior=0, **kwargs):
        self.board = board
        self.move = move
        self.is_expanded = False
        self.parent = parent  # Optional[UCTNode]
        self.children = OrderedDict()  # Dict[move, UCTNode]
        self.prior = prior  # float
        if parent is None or parent.parent is None:
            self.total_value = FPU_ROOT  # float
        else:
            self.total_value = FPU
        self.number_visits = 0  # int

    def Q(self):  # returns float
        return self.total_value / (1 + self.number_visits)

    def U(self):  # returns float
        return (
            math.sqrt(self.parent.number_visits) * self.prior / (1 + self.number_visits)
        )

    def best_child(self, C):
        return max(self.children.values(), key=lambda node: node.Q() + C * node.U())

    def select_leaf(self, C):
        current = self
        while current.is_expanded and current.children:
            current = current.best_child(C)
        if not current.board:
            current.board = current.parent.board.copy()
            current.board.push_uci(current.move)
        return current

    def expand(self, child_priors):
        self.is_expanded = True
        for move, prior in child_priors.items():
            self.add_child(move, prior)

    def add_child(self, move, prior):
        self.children[move] = UCTNode(parent=self, move=move, prior=prior)

    def backup(self, value_estimate: float):
        current = self
        # Child nodes are multiplied by -1 because we want max(-opponent eval)
        turnfactor = -1
        while current.parent is not None:
            current.number_visits += 1
            current.total_value += value_estimate * turnfactor
            current = current.parent
            turnfactor *= -1
        current.number_visits += 1


def rel_entropy_value(Q, p):
    return np.log(p.dot(np.exp(Q)))


def rel_entropy_max(Q, p):
    denominator = p.dot(np.exp(Q))
    return p * np.exp(Q) / denominator


def logit(p):
    return -np.log(1 / p - 1)


def expit(x):
    return 1 / (1 + np.exp(-x))


class RENTSNode(UCTNode):
    def __init__(
        self,
        board=None,
        parent=None,
        move=None,
        prior=0,
        discount_factor=1.0,
        eps=0.05,
        prior_visits=1,
        value_prior_strength=0.01,
        maximum_exploration=1.0,
        init_strategy="VplusP",
        fpu_value=0.0,
        fpu_strategy="reduction",
        fpu_value_at_root=0.1,
        fpu_strategy_at_root="absolute",
        **kwargs
    ):
        super().__init__(board=board, parent=parent, move=move, prior=prior, **kwargs)
        self.discount_factor = discount_factor
        self.eps = eps
        self.policy = prior
        self.prior_visits = prior_visits
        self.maximum_exploration = maximum_exploration
        self.init_strategy = init_strategy
        self.fpu_strategy = fpu_strategy
        self.fpu_strategy_at_root = fpu_strategy_at_root
        # self.total_value = -np.log(1/min(max(prior, 0.01), 0.99) - 1) * value_prior_strength
        if init_strategy == "fpu":
            is_root = parent is None or parent.parent is None
            if is_root:
                fpu = fpu_value_at_root
            else:
                fpu = fpu_value
            if (is_root and self.fpu_strategy_at_root == "reduction") or (
                not is_root and self.fpu_strategy == "reduction"
            ):
                self.total_value = -fpu
            else:
                self.total_value = fpu
        elif init_strategy == "VplusP" or init_strategy == "logP":
            self.initial_value = 0

    def add_child(self, move, prior):
        self.children[move] = RENTSNode(parent=self, move=move, prior=prior)

    def best_child(self, C):
        n_children = len(self.children)
        if n_children == 1:
            child = list(self.children.values())[0]
            return child
        p = [child.policy for child in self.children.values()]
        return list(self.children.values())[
            int(np.random.choice(n_children, size=1, p=p))
        ]

    def update_value(self, value, is_leaf):
        is_root = self.parent is None or self.parent.parent is None
        if is_leaf:
            if (
                (
                    (is_root and self.fpu_strategy_at_root == "reduction")
                    or (not is_root and self.fpu_strategy == "reduction")
                )
                or self.init_strategy == "logP"
                or self.init_strategy == "VplusP"
            ):
                self.total_value += value
            else:
                self.total_value = value
        else:
            self.total_value += value

    def update_policy(self, Q, p, visits):
        n_children = len(self.children)
        sum_visits = np.sum(visits)
        if sum_visits == 0:
            lambda_s = 1.0
        else:
            lambda_s = np.clip(
                self.eps * n_children / np.log(sum_visits + self.prior_visits), 0.0, 1.0
            )
        lambda_s = min(self.maximum_exploration, lambda_s)
        max_arg = rel_entropy_max(Q, p)
        new_p = (1 - lambda_s) * max_arg + lambda_s / n_children
        new_p /= new_p.sum()
        for i, child in enumerate(self.children.values()):
            child.policy = new_p[i]

    def backup(self, value_estimate: float):
        current = self
        if self.init_strategy == "VplusP":
            value_estimate = np.clip(value_estimate, -1 + 1e-10, 1 - 1e-10)
            value_estimate = np.arctanh(value_estimate)
            self.initial_value = value_estimate
            n_children = len(self.children)
            for child in self.children.values():
                prior = min(max(child.prior, 0.001), 0.999)
                child.total_value = self.initial_value + np.arctanh(prior - 1 / n_children)
        elif self.init_strategy == "logP":
            value_estimate = np.clip(value_estimate, -1 + 1e-10, 1 - 1e-10)
            value_estimate = np.arctanh(value_estimate)
            for child in self.children.values():
                prior = min(max(child.prior, 0.001), 0.999)
                child.total_value = np.log(prior)
        # print("\nValue estimate: {}".format(value_estimate))
        while current.parent is not None:
            current.number_visits += 1
            val = -value_estimate * current.discount_factor
            current.update_value(val, is_leaf=self == current)
            current = current.parent
            Q, p, visits = np.array(
                [
                    (child.Q(), child.policy, child.number_visits)
                    for child in current.children.values()
                ]
            ).T
            value_estimate = rel_entropy_value(Q, p)
            current.update_policy(Q, p, visits)
        current.number_visits += 1


def get_best_move(root):
    if isinstance(root, RENTSNode):
        bestmove, node = max(
            root.children.items(), key=lambda item: (item[1].policy, item[1].Q())
        )
    else:
    bestmove, node = max(
        root.children.items(), key=lambda item: (item[1].number_visits, item[1].Q())
    )
    score = int(round(cp(np.tanh(node.Q())), 0))
    return bestmove, node, score


def send_info(send, bestmove, count, delta, score):
    if send is not None:
        send(
            "info depth 1 seldepth 1 score cp {} nodes {} nps {} pv {}".format(
                score, count, int(round(count / delta, 0)), bestmove
            )
        )


def UCT_search(
    board,
    num_reads,
    net=None,
    C=1.0,
    verbose=False,
    max_time=None,
    tree=None,
    send=None,
):
    if max_time is None:
        # search for a maximum of an hour
        max_time = 3600.0
    max_time = max_time - 0.05

    start = time()
    count = 0
    delta_last = 0

    root = RENTSNode(board)
    for i in range(num_reads):
        count += 1
        leaf = root.select_leaf(C)
        child_priors, value_estimate = net.evaluate(leaf.board)
        leaf.expand(child_priors)
        leaf.backup(value_estimate)
        now = time()
        delta = now - start
        if delta - delta_last > 5:
            delta_last = delta
            bestmove, node, score = get_best_move(root)
            send_info(send, bestmove, count, delta, score)

        if (time is not None) and (delta > max_time):
            break

    bestmove, node, score = get_best_move(root)
    if send is not None:
        root.best_child(1.0)
        for nd in sorted(root.children.items(), key=lambda item: item[1].policy):
            send(
                "info string {} {} \t(P: {}%) \t (Pol: {}%) \t(Q: {})".format(
                    nd[1].move,
                    nd[1].number_visits,
                    round(nd[1].prior * 100, 2),
                    round(nd[1].policy * 100, 2),
                    round(np.tanh(nd[1].Q()), 5),
                )
            )
        send(
            "info depth 1 seldepth 1 score cp {} nodes {} nps {} pv {}".format(
                score, count, int(round(count / delta, 0)), bestmove
            )
        )

    # if we have a bad score, go for a draw
    return bestmove, score
