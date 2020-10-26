# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            counter = self.values.copy()
            for currState in self.mdp.getStates():
                if self.mdp.isTerminal(currState):
                    counter[currState] = 0
                else:
                    valuelist = []
                    for currAction in self.mdp.getPossibleActions(currState):
                        valuelist.append(self.computeQValueFromValues(currState, currAction))
                    counter[currState] = max(valuelist)
            self.values = counter

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        statesAndProb = self.mdp.getTransitionStatesAndProbs(state, action)
        sum = 0
        for i in statesAndProb:
            sum += i[1] * (self.mdp.getReward(state, action, i[0]) + self.discount * self.values[i[0]])
        return sum


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.mdp.getPossibleActions(state)
        if len(actions) == 0:
            return None
        bestAction = actions[0]
        maxQValue = -9999999999
        for i in actions:
            currValue = self.computeQValueFromValues(state, i)
            if currValue > maxQValue:
                maxQValue = currValue
                bestAction = i
        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)




    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        index = 0
        for i in range(self.iterations):
            if index == len(self.mdp.getStates()):
                index = 0
            counter = self.values.copy()
            currState = self.mdp.getStates()[index]
            if self.mdp.isTerminal(currState):
                index += 1
                continue
            else:
                valuelist = []
                for currAction in self.mdp.getPossibleActions(currState):
                    valuelist.append(self.computeQValueFromValues(currState, currAction))
                counter[currState] = max(valuelist)
            self.values = counter
            index += 1

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predecessors = {}
        queue = util.PriorityQueue()
        for i in self.mdp.getStates():
            predecessors[i] = []
        for i in self.mdp.getStates():
            if not self.mdp.isTerminal(i):
                #updating queue
                valuelist = []
                for currAction in self.mdp.getPossibleActions(i):
                    valuelist.append(self.computeQValueFromValues(i, currAction))
                diff = abs(self.values[i] - max(valuelist))
                queue.push(i, -diff)
                #updating predecessors
                for a in self.mdp.getPossibleActions(i):
                    for k in self.mdp.getTransitionStatesAndProbs(i, a):
                        if predecessors[k[0]].count(i) == 0:
                            predecessors[k[0]].append(i)

        for i in range(self.iterations):
            if queue.isEmpty():
                break
            s = queue.pop()
            valuelist = []
            for currAction in self.mdp.getPossibleActions(s):
                valuelist.append(self.computeQValueFromValues(s, currAction))
            self.values[s] = max(valuelist)

            for p in predecessors[s]:
                valuelist = []
                for currAction in self.mdp.getPossibleActions(p):
                    valuelist.append(self.computeQValueFromValues(p, currAction))
                diff = abs(self.values[p] - max(valuelist))
                if diff > self.theta:
                    queue.update(p, -diff)


