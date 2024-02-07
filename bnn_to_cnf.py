import math
from cnf import * 

def sequential_counter(l, D, SQ_id):
  ''' l is a list of Term objects. D is a constant. This returns
  a Cnf object with the constraint sum(l) >= D
  '''
  m = len(l)
  r = []
  cnf_clauses = Cnf([])

  if D < 1:
    term = Term(annotation='%s:R0-0' % (SQ_id))
    return (term, Cnf([Clause([term])]))
  elif m < D:
    term = Term(annotation='%s:R0-0' % (SQ_id))
    return (term, Cnf([Clause([term.neg()])]))

  for i in range(m):
    r.append([])
    for j in range(D+1):
      r[i].append(Term(annotation='%s:R%d-%d' % (SQ_id, i, j)))
  
  cnf_clauses += equiv(l[0], r[0][1])
  for j in range(2, D+1):
    cnf_clauses += Cnf([Clause([r[0][j].neg()])])
  for i in range(1,m):
    cnf_clauses += Cnf([
      Clause([r[i][1].neg(), l[i], r[i-1][1]]),
      Clause([r[i][1], l[i].neg()]),
      Clause([r[i][1], r[i-1][1].neg()])])
  
  for i in range(1,m):
    for j in range(2, D+1):
      cnf_clauses += Cnf([
        Clause([r[i][j].neg(), l[i], r[i-1][j]]),
        Clause([r[i][j].neg(), r[i-1][j-1], r[i-1][j]]),
        Clause([r[i][j], l[i].neg(), r[i-1][j-1].neg()]),
        Clause([r[i][j], r[i-1][j].neg()])])

  #print("Sequential counter: %d %d -> %d" % (m, D, len(cnf_clauses.clauses)) )

  return (r[m-1][D], cnf_clauses)

def internal_layer_to_cnf(x, a,bias, layer_id):
  #print('x length:', len(x))
  print('SHAPE :', a.shape)
  #print(type(a))
  #print(a)
  #print('Type of a:', type(a))
  #print('Content of a[0]:', a[0])
  print('Length of a[0]:', len(a[0]))



  #print('x :',x)
  #print('Content of layer.get_weights()[0]  :',a)
  si = len(x)
  so = len(a)
  print('si :', si)
  print('so :',so)
  # a= layer.get_weights()[0]
  assert len(a.shape) == 2, 'weights matrix should be 2D'
  #print('LENGTH :', a[0])
  assert( si == len(a[0]) ), 'input lengths do not match!'
  assert( so == len(bias) ), 'output lengths do not match!'

  output_terms = []
  cnf_clauses = Cnf([])

  for i in range(so):
    l = [None for _ in range(si)]
    sum_a_pos = 0
    sum_a_neg = 0
    for j in range(si):
      if a[i][j] == 1:
        l[j] = x[j]
        sum_a_pos += 1
      else:
        assert( a[i][j] == -1 ), 'invalid input'
        #print('x[j] :',x[j] )
        l[j] = x[j].neg()
        sum_a_neg -= 1

    C = bias[i]
    #print('C/2 :', C/2)

    D = (int)(math.ceil(-C/2.0 + (sum_a_pos+sum_a_neg)/2.0)) + abs(sum_a_neg)
    (final_term, clauses) = sequential_counter(l, D, '%s:S%d' % (layer_id, i))

    #if comp[i] == "<":
     # final_term = final_term.neg()

    output_terms.append(final_term)
    cnf_clauses += clauses

    #print(i, len(cnf_clauses.clauses))
    #print('output_terms : ',output_terms)
  
  return (output_terms, cnf_clauses)


def output_layer_to_cnf(x, a, bias,  layer_id):
  si = len(x)
  so = len(a[0])

  print('si :', si)
  print('so :',so)
  #print('Weights Softmax layer : ', a)
  #print('SHAPE :', a.shape)
  #print('a[0] and si  :', a[0], si)
  print('len(bias) and so  :', len(bias), so)

  assert( si == len(a[0]) ), 'input lengths do not match!'
  assert(so == len(bias)), 'output lengths do not match!'

  d = [[] for _ in range(so)]
  output_terms = []
  cnf_clauses = Cnf([])

  for i in range(so):
    for j in range(so):
      l = []
      sum_a_i = 0
      sum_a_j = 0
      sum_a_pos = 0
      sum_a_neg = 0
      for k in range(si):
        sum_a_i += a[i][k]
        sum_a_j += a[j][k]
        if a[i][k] == 1 and a[j][k] == -1:
          l.append(x[k])
          sum_a_pos += 1
        elif a[i][k] == -1 and a[j][k] == 1:
          l.append(x[k].neg())
          sum_a_neg -= 1

      bi, bj = -1*bias[i], -1*bias[j]
      E = math.ceil( (bj - bi + sum_a_i - sum_a_j) / 2.0 )
      D = (int)(math.ceil(E/2.0)) + abs(sum_a_neg)
      (final_term, clauses) = sequential_counter(l, D, '%s:S%d-%d' % (layer_id, i, j))
      d[i].append(final_term)
      cnf_clauses += clauses
      #print(f'i: {i}, j: {j}, k: {k}')
    
    (final_term, clauses) = sequential_counter(d[i], so, '%s:S%d' % (layer_id, i))
    output_terms.append(final_term)
    cnf_clauses += clauses
    #print('output_terms :',output_terms)
    
  return (output_terms, cnf_clauses)


def sign(x):
  return 1 if x >= 0 else -1
