import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
from enum import Enum, auto
import sys
import time
import torch
import torch.optim as optim

#generating masks for results
#try a bigger example/sample mask each time
#try with deeper tensors
#try with samples from the language?

torch.set_default_tensor_type(torch.FloatTensor)


start = time.time()

class LanguageFrame:
	def __init__(self, target, Pe, arity_e, C):
		self.target = target
		self.Pe = Pe
		self.arity_e = arity_e # Pe U Target -> N
		self.C = C

class ILP_Problem:
	def __init__(self, LangFrame, Background_A, Pos, Neg):
		self.LangFrame = LangFrame
		self.Background_A = Background_A
		self.Pos = Pos # Pe U Target -> N
		self.Neg = Neg

class Atom:
	def __init__(self, predicate, terms):
		self.predicate = predicate
		self.terms = terms
		self.ground = False
	def __eq__(self, other):
	    return self.terms == other.terms and self.predicate == other.predicate

class Definite_clause:
	def __init__(self,head,body):
		self.head = head # atom
		self.body = body # list of atoms

class Ground_atom:
	def __init__(self, predicate, terms):
		self.predicate = predicate
		self.terms = terms
		self.ground = True

class Rule_Template:
	def __init__(self, v, inte):
		self.v = v #in N, num of existens quant vars in clause
		self.inte = inte # flag 0 or 1 whether atoms can use intens preds

class Program_Template:
	def __init__(self, Pa, arity_a, rules, T):
		self.Pa = Pa #invented predicates
		self.arity_a = arity_a # arity of invented predicates
		self.rules = rules #gives each Pa two rule templates
		self.T = T # int, max number of steps of forward chaining inference

class Language:
	def __init__(self,Pe,Pi,arity,C):
		self.Pe = Pe
		self.Pi = Pi
		self.arity = arity
		self.C = C

def FrameAndTemp_ToLang(langFrame, Template):
	return Language(langFrame.Pe,Template.Pa + [langFrame.target],langFrame.arity_e+Template.arity_a,langFrame.C)

def generate_predTemp_cl(predicate,rule_template, pred_arity, lang):
	P = lang.Pe + lang.Pi
	re_arity = lang.arity[0:len(lang.Pe)] + lang.arity[len(lang.Pe)+1:]+ [lang.arity[len(lang.Pe)]]

	#all the options for one template
	clause_list = []
	bodyatomList = []
	var_list = []	

	for v in range(0,pred_arity + rule_template.v): #+1
		var_list.append('x'+str(v+1))
	# print(var_list)
	exvar_list = var_list[pred_arity:]
	# print(exvar_list)
	univar_list = var_list[:pred_arity]
	head = Atom(predicate,univar_list)
	# print(['a','b'] + 'b')

	typvar = univar_list+['bl']
	# ['x1','bl']
#=========
	# print('abc'<'abd')
	for ind in range(0,len(P)):
		pred = P[ind]
		ar  = re_arity[ind]
		for ind2 in range(0,len(P)):

			pred2 = P[ind2]
			ar2  = re_arity[ind2]
			totalar = ar +ar2
			if pred>=pred2:
				#we have total ar
				# we have typvar
				numtyps = len(typvar)
				termchoices = []
				totalOptNum = (len(typvar)**totalar)
				for L in range(0,totalOptNum):
					# the l'th choice
					currchoice = []
					for p in range(0,totalar):
						currchoice.append(typvar[  (L//(numtyps**(totalar-p-1)))%numtyps  ])
					termchoices.append(currchoice)
				exchoices = []
				#have termchoices in terms of universal variables and one exist var
				#to get in terms of n eist vars - take each option, replace first
				for termchoice in termchoices:
					replace_steps = 0
					currentopts = [termchoice]
					for teInd in range(0,len(termchoice)):
						if termchoice[teInd] == 'bl':
							new_currentopts = []
							for opt in currentopts:
								for exvarInd in range(0,min(replace_steps+1,len(exvar_list))):
									changedOpt = opt
									changedOpt[teInd] = exvar_list[exvarInd]
									new_currentopts.append(changedOpt)
							currentopts = new_currentopts
							replace_steps = replace_steps+1
					exchoices = exchoices + currentopts
				tricky = []
				for fullchoice in exchoices:
					contains = True
					for uVar in head.terms:
						found = False
						for inVar in fullchoice:
							if uVar == inVar:
								found = True
						if not found:
							contains = False
					if contains:

						if not((head.predicate == pred and fullchoice[:ar] == head.terms) or  (head.predicate == pred2 and fullchoice[ar:] == head.terms)):
							
							# if False:
							# 	pass
							if pred == pred2 :
								unpet = True
								if len(tricky)>0:
									for trick in tricky:
										if trick[0].predicate == pred2 and trick[1].predicate == pred and trick[0].terms == fullchoice[ar:] and trick[1].terms == fullchoice[:ar]:
											unpet = False
								if unpet:
									tricky.append([Atom(pred,fullchoice[:ar]),Atom(pred2,fullchoice[ar:])])
									bodyatomList.append([Atom(pred,fullchoice[:ar]),Atom(pred2,fullchoice[ar:])])
							else:
								bodyatomList.append([Atom(pred,fullchoice[:ar]),Atom(pred2,fullchoice[ar:])])
	finalList = []
	for bodyatom in bodyatomList:
		finalList.append(Definite_clause(head,bodyatom))
		# print(head.predicate,"(",head.terms,") <-",
		# print(bodyatom[0].predicate,"(",bodyatom[0].terms,"),",bodyatom[1].predicate,"(",bodyatom[1].terms,")")

	 #=============================================================

	# print(Atom('',['a','v'])==Atom('',['a','v']))

	#todo
	#delete if head is an atom in body
	# print(typvar)
	# print(univar_list)
	# print(exvar_list)
	# print(var_list)
	return finalList

def generate_all_ground_atoms(lang):
	P = lang.Pe + lang.Pi
	# p is pe u pi, pi is pa u t, so ... pe, pa , t
	# arity is ae u aa, ae is pe u t, aa is pa, so pe, t ,pe
	# print(lang.arity)
	# print(P)
	re_arity = lang.arity[0:len(lang.Pe)] + lang.arity[len(lang.Pe)+1:]+ [lang.arity[len(lang.Pe)]]
	# print(re_arity)
	G = []
	G.append(Ground_atom('F',[]))
	for ind in range(0,len(P)):
		# print(P[ind], re_arity[ind])
		pred = P[ind]
		ar = re_arity[ind]
		if ar == 0:
			G.append(Ground_atom(pred,[]))
		if ar == 1:
			for cons in lang.C:
				G.append(Ground_atom(pred,[cons]))
		if ar == 2:
			for cons in lang.C:
				for oCons in lang.C:
					G.append(Ground_atom(pred,[cons,oCons]))
	# for gr_a in G:
	# 	print(gr_a.predicate,gr_a.terms)
	return G


def X_index_creation(clause, groundAtomList, constants,Rule_Template ):
	# print(clause.head.predicate, clause.head.terms , clause.body[0].predicate, clause.body[0].terms,clause.body[1].predicate, clause.body[1].terms )
	# print(len(groundAtomList))
	# print(groundAtomList[len(groundAtomList)//2].predicate,groundAtomList[len(groundAtomList)//2].terms)
	v = Rule_Template.v
	w = len(constants)**v #max size of X
	n = len(groundAtomList)
	# print('w =',w,' n=',n)
	x = torch.zeros((n, w , 2),dtype = torch.long)
	# x = np.zeros((n, w , 2))
	# print(x)

	# interesting = False
	# if clause.head.predicate == 'succ2' and clause.body[0].predicate == 'succ'and clause.body[1].predicate == 'succ' and clause.body[0].terms == ['x1','x3'] and clause.body[1].terms == ['x3','x2']:
	# 	interesting = True

	for head_ind in range(0,n):
		worked = 0
		full = False
		
		# if interesting:
		# 	print(head_ind,worked,full)
		
		for body_atom_ind1 in range(0, n):
			for body_atom_ind2 in range(0,n):

				
				# if interesting and head_ind == 15 and body_atom_ind1 == 5 and body_atom_ind2 == 9:
				# 	print("\n$£$£$£\n")

				if any_sub_works(clause, groundAtomList[head_ind],groundAtomList[body_atom_ind1],groundAtomList[body_atom_ind2]):
					
					# if groundAtomList[head_ind].predicate=='even' and len(groundAtomList[head_ind].terms)==1 and groundAtomList[head_ind].terms[0]=='0':
					# 	print("FF",groundAtomList[body_atom_ind1].predicate,groundAtomList[body_atom_ind1].terms,groundAtomList[body_atom_ind2].predicate,groundAtomList[body_atom_ind2].terms)

					x[head_ind, worked, 0] = body_atom_ind1
					x[head_ind,worked,1] = body_atom_ind2
					worked = worked +1 

					#testfirst
					# full = True

					if worked >= w:
						full = True
				if full:
					break
			if full:
				break

	# if clause.head.predicate =='succ2' and clause.head.terms == ['x1','x2'] and clause.body[0].predicate == 'succ' and clause.body[0].terms == ['x1','x3'] and clause.body[1].predicate == 'succ' and clause.body[1].terms == ['x3','x2']:
	# 	print(x[:,:,0])
	# 	print(x[:,:,1])
	# if clause.head.predicate =='even' and clause.head.terms == ['x1'] and clause.body[0].predicate == 'succ2' and clause.body[0].terms == ['x2','x1'] and clause.body[1].predicate == 'even' and clause.body[1].terms == ['x2']:
	# 	print(x[:,:,0])
	# 	print(x[:,:,1])

	# print(clause.head.predicate, clause.head.terms,clause.body[0].predicate ,clause.body[0].terms ,clause.body[1].predicate,clause.body[1].terms)


	return x

def getListofTerms(clause):
	listofterms = []
	for term in clause.head.terms:
		if term not in listofterms:
			listofterms.append(term)
	for term in clause.body[0].terms:
		if term not in listofterms:
			listofterms.append(term)
	for term in clause.body[1].terms:
		if term not in listofterms:
			listofterms.append(term)
	return listofterms

def any_sub_works(clause, head, body1, body2):
	l1 = getListofTerms(clause)
	l2 = getListofTerms(Definite_clause(head,[body1,body2]))

	if clause.head.predicate != head.predicate:
		return False
	if not ((clause.body[0].predicate == body1.predicate and clause.body[1].predicate == body2.predicate)
		or (clause.body[1].predicate == body1.predicate and clause.body[0].predicate == body2.predicate)):
		return False

	if len(l2)!= len(l1) and (len(l1) == 0 or len(l2) == 0):
		return False

	# print(l1,l2)
	subs = []
	for vartosub in l1:
		varsublist = []
		for poss in l2:
			varsublist.append([vartosub,poss])
		subs.append(varsublist)
	# print(subs)

	going = True
	first = True
	choices_inds = []
	for listy in subs:
		choices_inds.append(0)
	choices_inds[len(choices_inds)-1]=-1
	while going == True:
		
		#check for all checked
		allzeroes = True
		for id in choices_inds:
			if id != 0:
				allzeroes = False 
		if allzeroes ==True:
			if first ==True:
				first = False
			else:
				going = False
				break

		for rev_ind in range(0,len(choices_inds)):

			ind = len(choices_inds)-1-rev_ind

			if choices_inds[ind] == len(subs[ind])-1:
				choices_inds[ind]=0
			else: 
				choices_inds[ind] = choices_inds[ind]+1
				break

		# now we check the individual sub, choices_ind has a list of which ind you need to look at in subs
		# go thru choices_ind, for each ind, find the var sub referenced in subs, take each sub tuple and go thru
		# the head and  body atoms and swap each var for c, then see if it matches clause, rememeber to check atoms
		# in both orders!!!
		subtups = []
		for chosensubind in range(0,len(choices_inds)):
			# print(chosensubind,'@',choices_inds[chosensubind])
			subtups.append(subs[chosensubind][choices_inds[chosensubind]])
		# print(subtups)
		oldheadterms = clause.head.terms
		oldbodyAterms = clause.body[0].terms
		oldbodyBterms = clause.body[1].terms
		head_terms = []
		for term in oldheadterms:
			for posub in subtups:
				if posub[0] == term:
					head_terms.append(posub[1])

		bodyAterms = []
		for term in oldbodyAterms:
			for posub in subtups:
				if posub[0] == term:
					bodyAterms.append(posub[1])
		bodyBterms = []
		for term in oldbodyBterms:
			for posub in subtups:
				if posub[0] == term:
					bodyBterms.append(posub[1])

		working_head = (clause.head.predicate == head.predicate) and head.terms == head_terms

		working_body_1 = ((clause.body[0].predicate == body1.predicate) and (clause.body[1].predicate == body2.predicate )and(
			body1.terms == bodyAterms and body2.terms == bodyBterms)) 

		working_body_2 = ((clause.body[0].predicate == body2.predicate) and (clause.body[1].predicate == body1.predicate )and(
			body2.terms == bodyAterms and body1.terms == bodyBterms)) 
		if working_head and (working_body_1 or working_body_2):
			return True


		# print("")
		# going = False
	return False
# def X_index_creation(valuations, clause, groundAtomList):
# 	print(valuations)
# 	print(clause.head.predicate, clause.head.terms , clause.body[0].predicate, clause.body[0].terms,clause.body[1].predicate, clause.body[1].terms )
# 	print(len(groundAtomList))
# 	print(groundAtomList[91].predicate,groundAtomList[91].terms)

#####====================================Program creation
#####====================================Program creation
#####====================================Program creation
#####====================================Program creation
#####====================================Program creation
#####====================================Program creation
#####====================================Program creation
#making a language frame test
Pe_test = ['zero','succ']
target_test = 'even'
arity_e_test = [1,2,1] # Pe U Target -> N
C_test = ['0','1','2','3','4','5','6','7','8']
c_baby = ['0','1','2','3','4']
C_test = c_baby
langF_test = LanguageFrame(target_test,Pe_test,arity_e_test,C_test)

#making an ILP problem spec test
Background_A_test = [Ground_atom('zero',['0'])]
for x in range(0,3):
	Background_A_test.append(Ground_atom('succ',[C_test[x],C_test[x+1]]))
# # pos_test = []
# neg_test = []
pos_test = ['0','2','4']#,'6','8']
neg_test = ['1','3']#,'5','7']

# pos_test = ['2']
# neg_test = ['0','1']

ILP_Problem_test = ILP_Problem(langF_test,Background_A_test,pos_test,neg_test)

#making a test program template
Pa_test = ['succ2']
arity_a_test = [2]
T_test = 4
Rule_Template_test = Rule_Template(1,1)
rules_test = [[Rule_Template_test,Rule_Template_test]]
Program_Template_test = Program_Template(Pa_test,arity_a_test,rules_test,T_test)


Language_test = FrameAndTemp_ToLang(langF_test,Program_Template_test)

G_test = generate_all_ground_atoms(Language_test)
u = 0
for gr_a in G_test:
	print(u,gr_a.predicate,gr_a.terms)
	u=u+1
print(len(G_test), "ground atoms generated")

testgen = generate_predTemp_cl(Program_Template_test.Pa[0],Program_Template_test.rules[0][0], Program_Template_test.arity_a[0],Language_test)
# for clause in testgen:
# 	print(clause.head.predicate,"(",clause.head.terms,") <-",clause.body[0].predicate,"(",clause.body[0].terms,"),",clause.body[1].predicate,"(",clause.body[1].terms,")")
print(len(testgen),"clauses generated for predicate ",Program_Template_test.Pa[0], " with arity ",Program_Template_test.arity_a[0], "for its first rule temlplate")

valuations_Test = [0]
for u in range(1,len(G_test)):
	if u%13 == 0 or u%7 == 0 or u%11 == 0 or u%8 == 0:
		valuations_Test.append(1)
	else:
		valuations_Test.append(0)

# for t in range(0,100):
X_index_creation(testgen[0], G_test,C_test, Rule_Template_test)
#takes around 0.2 seconds per clause currently, so with 2-300 clauses*2 takes about 150 seconds for full gen

#####====================================End









			

def whole_shebang(ILP_prob,Program_temp):
	start = time.time()

	#ILP_prob has languageframe, background atoms, positive and negative examples
	#program template has the list of auxilliary preds, the arity for these, the rule templates, and the steps of forward inferene t
#problem setting========================
	lang = FrameAndTemp_ToLang(ILP_prob.LangFrame,Program_temp)
	all_ground_atoms = generate_all_ground_atoms(lang)
	print(len(all_ground_atoms), "ground atoms generated")
	clause_gen = [[generate_predTemp_cl(ILP_prob.LangFrame.target, Rule_Template(1,1) ,ILP_prob.LangFrame.arity_e[len(ILP_prob.LangFrame.Pe)],lang)]*2]
	for prin in range(0,len(Program_temp.Pa)):
		pred = Program_temp.Pa[prin]
		clause_gen.append([generate_predTemp_cl(pred,Program_temp.rules[prin][0],Program_temp.arity_a[prin],lang),generate_predTemp_cl(pred,Program_temp.rules[prin][1],Program_temp.arity_a[prin],lang)])
	cll = clause_gen[1][0][40]
	print(cll.head.predicate,cll.head.terms,cll.body[0].predicate,cll.body[0].terms,cll.body[1].predicate,cll.body[1].terms)
	cll = clause_gen[0][0][55]
	print(cll.head.predicate,cll.head.terms,cll.body[0].predicate,cll.body[0].terms,cll.body[1].predicate,cll.body[1].terms)
	cll = clause_gen[0][0][0]
	print(cll.head.predicate,cll.head.terms,cll.body[0].predicate,cll.body[0].terms,cll.body[1].predicate,cll.body[1].terms)
	print(clause_gen[0][1][10].head.predicate,clause_gen[0][1][10].head.terms,clause_gen[0][1][10].body[0].predicate,clause_gen[0][1][10].body[0].terms,clause_gen[0][1][10].body[1].predicate,clause_gen[0][1][10].body[1].terms)

	a0 = []
	for gr_atom in all_ground_atoms:
		found = False
		for back in ILP_prob.Background_A:
			if gr_atom.predicate == back.predicate and gr_atom.terms == back.terms:
				a0.append(1)
				found = True
				break
		if not found:
			a0.append(0)
	print(a0)
	# np.random.seed(0)
	torch.manual_seed(2)
	weights = []
	for pred in clause_gen: 
		# weights.append(np.random.normal(size = (len(pred[0]),len(pred[1]))))#/((len(pred[0])*len(pred[1]))**2))
		weights.append(torch.randn(size = (len(pred[0]),len(pred[1])),requires_grad =True))#/((len(pred[0])*len(pred[1]))**2))


	# print(weights)
	# print(weights[0])
	# print(weights[0].shape)
	# return

	current_a = torch.tensor(a0,dtype = torch.float32)
	# current_a = np.array(a0)
	print("loss without workings",calcLoss(current_a,ILP_prob.Pos,ILP_prob.Neg,ILP_prob.LangFrame.target,all_ground_atoms,True))
	#generate xs

	end = time.time()
	print("generation took:",end-start)
	sys.stdout.flush() 
	start = time.time()

	print("performing function pre-processing:")
	sys.stdout.flush() 

	xs = []
	for clg_in in range(0,len(clause_gen)):
		intx = []
		for cltem_in in range(0,len(clause_gen[clg_in])): # 0or 1
			intc = []
			for clause_index in range(0,len(clause_gen[clg_in][cltem_in])):
				#TODO HERE I USED Rule_Template(1,1) INSTEAD OF THE REAL THING BAAADD
				ruletemp = Rule_Template(1,1)
				if clg_in>0:
					ruletemp = Program_temp.rules[cltem_in-1][cltem_in]
				mcl = clause_gen[clg_in][cltem_in][clause_index]
				intc.append( X_index_creation(clause_gen[clg_in][cltem_in][clause_index], all_ground_atoms, ILP_prob.LangFrame.C, ruletemp ) )
				print(clause_index,"/",len(clause_gen[clg_in][cltem_in]))
				sys.stdout.flush()
			intx.append(intc)
		print("\n\n",clg_in,"/",len(clause_gen))
		sys.stdout.flush
		xs.append(intx)

	# print(xs)
	end = time.time()
	print("X creation took:",end-start)
	sys.stdout.flush() 


#============================================= optimisation

	# optimizer = optim.SGD(weights, lr = 0.1, momentum = 0.9)
	optimizer = optim.Adam(weights, lr = 0.2)


	for t in range(0,30):
		start = time.time()

		optimizer.zero_grad()


		loss = to_loss(weights,Program_temp,current_a,clause_gen,xs,ILP_prob,all_ground_atoms,True)
		print("\n\npass",t," loss ",loss)

		end = time.time()
		print("loss calculation took:",end-start)
		sys.stdout.flush() 
		start2 = time.time()

		loss.backward()
		
		end = time.time()
		print("backward took:",end-start2)
		sys.stdout.flush() 

		optimizer.step()

		print("whole step took", time.time()-start)
		# with torch.no_grad():
		# 	for wm in range(0,len(weights)):
		# 		weights[wm] -= 0.1* weights[wm].grad
		# 		weights[wm].grad.zero_()



	loss = to_loss(weights,Program_temp,current_a,clause_gen,xs,ILP_prob,all_ground_atoms,True)
	print("\n\n final loss ",loss)



	return -1

	# gradloss = grad(to_loss,0)

	# end = time.time()
	# print("grad function calculation took:",end-start)
	# sys.stdout.flush() 
	# start = time.time()

	# for iterations in range(0,800):
	# 	print("\nITERATION ",iterations)
	# 	wgrad = gradloss(weights,Program_temp,current_a,clause_gen,xs,ILP_prob,all_ground_atoms,False)
		
	# 	end = time.time()
	# 	print("backward pass took:",end-start)
	# 	sys.stdout.flush() 
	# 	start = time.time()

	# 	for bp in wgrad:
	# 		print("maxgrad:",np.max(bp.flatten()))
	# 	for wm in range(0,len(weights)):
	# 		if iterations<3:
	# 			weights[wm] = weights[wm] -  wgrad[wm]*20
	# 		# elif iterations<7:
	# 		# 	weights[wm] = weights[wm] -  wgrad[wm]*10
	# 		elif iterations<10:
	# 			weights[wm] = weights[wm] -  wgrad[wm]*5
	# 		else:
	# 			weights[wm] = weights[wm] -  wgrad[wm]*0.1

	# 	end = time.time()
	# 	print("weight update took:",end-start)
	# 	sys.stdout.flush() 
	# 	start = time.time()

	# 	print("\nforward new loss",to_loss(weights,Program_temp,current_a,clause_gen,xs,ILP_prob,all_ground_atoms,True))

	# 	end = time.time()
	# 	print("forward calculation took:",end-start)
	# 	sys.stdout.flush() 
	# 	start = time.time()

	# return -1


# def softmaxw(op1_ind,op2_ind,w):
# 	wf = w.flatten()
# 	top = w[op1_ind][op2_ind] - np.max(wf)
# 	return np.exp(top)/(np.sum(np.exp(wf-np.max(wf))))

# def softmaxfnp(w):
# 	wf = w.flatten()
# 	top = w- np.max(wf)
# 	return np.exp(top)/(np.sum(np.exp(wf-np.max(wf))))


def softmaxf(w):
	wf = torch.flatten(w)
	top = w- torch.max(wf)
	return torch.exp(top)/(torch.sum(torch.exp(wf-torch.max(wf))))

def gather_and_con(a,x):
	x1  = x[:,:,0]
	x2  = x[:,:,1]
	# print(x1)
	# print(a)
	y1 =  a[x1]#.astype(int)]
	y2 =  a[x2]#.astype(int)]
	# z = np.multiply(y1,y2)
	# f = np.amax(z,axis = 1)	
	z = y1*y2
	f = torch.max(z, 1)[0]
	# print("!",f)
	return f

def calcLoss(a_n,pos,neg,target,groundAtomList,forward):
	total_negloss = 0
	for at_ind in range(0,len(groundAtomList)):
		gr_At = groundAtomList[at_ind]
		if gr_At.predicate == target:
			#only works for unary target TODO
			# print(a_n[at_ind])
			# print((a_n[at_ind]+0.00001))

			if gr_At.terms[0] in pos:
				term = torch.log(a_n[at_ind]+0.00001)
				# term = np.log(a_n[at_ind]+0.00001)
				if forward: print(term)
			else: 
				term = torch.log(1-a_n[at_ind] + 0.00001)
				# term = np.log(1-a_n[at_ind] + 0.00001)
				if forward: print(term)
			total_negloss = total_negloss + term
	return - total_negloss

def to_loss(weights,Program_temp,current_a,clause_gen,xs,ILP_prob,all_ground_atoms, forward):
	s = time.time()

	if forward: print(current_a)	
	for step in range(1,Program_temp.T+1): 
		sump = torch.zeros((len(current_a)))
		for pred_ind in range(0,len(clause_gen)): #pa/t calculate b for each pa
			softmaxweightsfull = softmaxf(weights[pred_ind])
			# print("AaAa",softmaxweightsfull.shape)
			softmaxweightsre = softmaxweightsfull.view(softmaxweightsfull.shape + (1,))
			# print(softmaxweightsre.shape)
			# a_throughf = torch.tensor([[torch.max((torch.tensor([gather_and_con(current_a,xs[pred_ind][0][op1_ind]),gather_and_con(current_a,xs[pred_ind][1][op2_ind])])), 0) for op2_ind in range(0,len(clause_gen[pred_ind][1]))] for op1_ind in range(0,len(clause_gen[pred_ind][0]))])
			

			# a_throughf = torch.tensor(
			a_throughf =             torch.stack([torch.stack([torch.max(torch.stack(  [  gather_and_con(current_a,xs[pred_ind][0][op1_ind]),gather_and_con(current_a,xs[pred_ind][1][op2_ind])  ]  ), 0)[0] for op2_ind in range(0,len(clause_gen[pred_ind][1]))]) for op1_ind in range(0,len(clause_gen[pred_ind][0]))])#)
			# print(a_throughf.shape)
			# print(a_throughf)
			# a_throughf = [(gather_and_con(current_a,xs[pred_ind][0][op1_ind])) for op1_ind in range(0,4)]
			
			# tosd.asdf()


			# a_throughf = np.array([[np.amax((np.array([gather_and_con(current_a,xs[pred_ind][0][op1_ind]),gather_and_con(current_a,xs[pred_ind][1][op2_ind])])),axis = 0) for op2_ind in range(0,len(clause_gen[pred_ind][1]))] for op1_ind in range(0,len(clause_gen[pred_ind][0]))])
			# Interm = np.multiply(a_throughf,softmaxweightsre).reshape(-1,len(current_a)).sum(axis = 0)
			
			Interm = torch.sum((a_throughf*softmaxweightsre).view(-1,len(current_a)),dim = 0)
			sump = sump + Interm
		current_a = current_a + sump - (current_a*sump)
		if forward: print("post ",step,current_a)
		e = time.time()
		print("(tstep",e-s,")")
		s = e
		sys.stdout.flush() 
	loss = calcLoss(current_a,ILP_prob.Pos,ILP_prob.Neg,ILP_prob.LangFrame.target,all_ground_atoms, forward)
	return loss



end = time.time()
print("problem set up took:",end-start)
sys.stdout.flush() 

whole_shebang(ILP_Problem_test,Program_Template_test)

