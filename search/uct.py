import math
from collections import OrderedDict
from time import time

import chess
import numpy as np
from search.util import cp

FPU = -1.0
FPU_ROOT = 0.0


class UCTNode(object):
    def __init__(self, board=None, parent=None, move=None, prior=0, **kwargs):
        self.board = board
        self.move = move
        self.is_expanded = False
        self.parent = parent  # Optional[UCTNode]
        self.children = OrderedDict()  # Dict[move, UCTNode]
        self.prior = prior  # float
        if parent is None:
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


class RENTSNode(UCTNode):

    def __init__(
        self,
        board=None,
        parent=None,
        move=None,
        prior=0,
        discount_factor=1.0,
        eps=0.0001,
        **kwargs
    ):
        super().__init__(board=board, parent=parent, move=move, prior=prior, **kwargs)
        self.discount_factor = discount_factor
        self.eps = eps
        self.policy = prior
        self.total_value = 0.0

    def add_child(self, move, prior):
        self.children[move] = RENTSNode(parent=self, move=move, prior=prior)

    def best_child(self, C):
        n_children = len(self.children)
        visits = [child.number_visits + 1 for child in self.children.values()]
        # sum_visits = np.sum(visits)
        # if sum_visits == 0:
        #     p = np.array([child.policy for child in self.children.values()])
        #     p /= p.sum()
        #     return list(self.children.values())[int(np.random.choice(n_children, size=1, p=p))]
        lambda_s = self.eps * n_children / np.log(np.sum(visits))
        Q, p = np.array(
            [(child.Q(), child.policy) for child in self.children.values()]
        ).T
        #print("lambda_s = {}".format(lambda_s))
        #print("Q = {}".format(Q))
        #print("p = {}".format(p))
        max_arg = rel_entropy_max(Q, p)
        new_p = (1 - lambda_s) * max_arg + lambda_s / n_children
        #print(new_p)
        if np.any(new_p < 0.0):
            print("lambda_s = {}".format(lambda_s))
            print("Q = {}".format(Q))
            print("p = {}".format(p))
            print("new_p = {}".format(new_p))
        new_p /= new_p.sum()
        for i, child in enumerate(self.children.values()):
            child.policy = new_p[i]
        return list(self.children.values())[int(np.random.choice(n_children, size=1, p=new_p))]

    def backup(self, value_estimate: float):
        current = self
        # Child nodes are multiplied by -1 because we want max(-opponent eval)
        #print("\nInitial estimate: {}".format(value_estimate))
        while current.parent is not None:
            current.number_visits += 1
            #print("Value before: {}".format(current.total_value))
            current.total_value += -value_estimate * self.discount_factor
            #print("Value after: {}".format(current.total_value))
            current = current.parent
            Q, p = np.array(
                [(child.Q(), child.policy) for child in current.children.values()]
            ).T
            #print("Q: {}".format(Q))
            #print("p: {}".format(p))
            value_estimate = rel_entropy_value(Q, p)
            #print("Value: {}".format(value_estimate))
        current.number_visits += 1


def get_best_move(root):
    bestmove, node = max(
        root.children.items(), key=lambda item: (item[1].number_visits, item[1].Q())
    )
    score = int(round(cp(node.Q()), 0))
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
        for nd in sorted(root.children.items(), key=lambda item: item[1].number_visits):
            send(
                "info string {} {} \t(P: {}%) \t (Pol: {}%) \t(Q: {})".format(
                    nd[1].move,
                    nd[1].number_visits,
                    round(nd[1].prior * 100, 2),
                    round(nd[1].policy * 100, 2),
                    round(nd[1].Q(), 5),
                )
            )
        send(
            "info depth 1 seldepth 1 score cp {} nodes {} nps {} pv {}".format(
                score, count, int(round(count / delta, 0)), bestmove
            )
        )

    # if we have a bad score, go for a draw
    return bestmove, score
