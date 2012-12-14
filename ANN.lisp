;; Bentley Cook
;; CS 356
;; Artificial Neural Network

;; To run XOR problem call magic with a list containing the lengths of the 
;; layers you want to create, the test input data, the max number of iterations
;; and the e used to check for convergance.
;; For example, (magic '(2 5 1) my-in 500 .15)
;; This algorithm only checks for convergance when the output error is below.
;; I began to implement the other checks but for the sake of time, stopped. However,
;; this algorithm seems to always converge correctly. Additionally, I did not implement
;; any 'fancy' networks for the XOR problem aside from ones with more or less than 3
;; hidden nodes.

;; To solve the number recognition algorithm, call recnumber with a list of the size
;; of the layers you want, fullowed by the test input, the expected output, max number
;; of iterations, the e to check against (although this isn't used to solve this problem),
;; and the MSE you want to check against.
;; For example, (recnumber '(35 10 10) numbers numbersOutput 1000 .15 .03)

;; (recnumber '(35 10 10) numbers numbersOutput 100000 .15 .001)
;; Convergence reached after 3324 iterations
;; The result for number 0 was 0 with a result of 0.96975696
;; The result for number 1 was 1 with a result of 0.98708516
;; The result for number 2 was 2 with a result of 0.97980666
;; The result for number 3 was 3 with a result of 0.9757791
;; The result for number 4 was 4 with a result of 0.98574084
;; The result for number 5 was 5 with a result of 0.9770711
;; The result for number 6 was 6 with a result of 0.9625868
;; The result for number 7 was 7 with a result of 0.968935
;; The result for number 8 was 8 with a result of 0.97634
;; The result for number 9 was 9 with a result of 0.97465366
;; Recnumber is the algorithm that solves the number problem. I somtimes could not get
;; it to converge with a MSE < .001. However, I got very successful results
;; having it converge with a MSE < .03.
;; (recnumber '(35 10 10) numbers numbersOutput 1000 .15 .03)
;; Convergence reached after 408 iterations
;; The result for number 0 was 0 with a result of 0.8955003
;; The result for number 1 was 1 with a result of 0.9417826
;; The result for number 2 was 2 with a result of 0.92305344
;; The result for number 3 was 3 with a result of 0.8564149
;; The result for number 4 was 4 with a result of 0.8970035
;; The result for number 5 was 5 with a result of 0.93287647
;; The result for number 6 was 6 with a result of 0.8513432
;; The result for number 7 was 7 with a result of 0.904715
;; The result for number 8 was 8 with a result of 0.9040851
;; The result for number 9 was 9 with a result of 0.9028278
;; recnumber : listOf numbers, listOf inputData, listOf expectedOutput, number, number, number -> NEURAL-NETWORK
(defun recnumber (net-sizes data ex-out max-iter e MSE-min)
  (setf *e* e)
  (setf *stop* NIL)
  (setf testnet (createnetwork net-sizes))
  (connectnet testnet)
  (loop for iter from 1 to max-iter do
    (loop for currentNum from 0 to 9 do
      ;;(format t "Beggining of iteration ~A on number ~A~%" iter currentNum)
      (setf data-in (loop for num across (nth currentNum data) collect num))
      ;;(format t "Forward-prop of iteration ~A on number ~A~%" iter currentNum)
      (forward-prop testnet data-in)
      ;;(format t "Back-prop of iteration ~A on number ~A~%" iter currentNum)
      (back-prop testnet (nth currentNum ex-out)))
    ;; Begin to check for convergence
    (setf MSE 
          (loop for out-node in (net-out-layer mynet) sum
            (* (node-error out-node) (node-error out-node))))
    ;;(format t "MSE: ~A~%" MSE)
    (when (< MSE MSE-min)
      (setf *stop* 'MSEReached)
      (setf finaliter iter)
      (setf iter (+ 1 max-iter))))
  (if *stop*
    (progn
      (format t "Convergence reached after ~A iterations~&" finaliter)
      (loop for currentNum from 0 to 9 do
        (setf large '(-1 -1))
        (setf data-in (loop for num across (nth currentNum data) collect num))
        (forward-prop testnet data-in)
        (loop for k from 0 to 9 do
          (let ((current (node-value (nth k (net-out-layer testnet)))))
            (when (> current (second large))
              (setf large (list k current)))))
        (format t "The result for number ~A was ~A with a result of ~A~%" currentNum (first large) (second large))))
    (progn
      (format t "Convergence was not reached after ~A iterations~%" max-iter)
      (loop for currentNum from 0 to 9 do
        ;; Large: (nodeNum nodeResult)
        (setf large '(-1 -1))
        (setf data-in (loop for num across (nth currentNum data) collect num))
        (forward-prop testnet data-in)
        (loop for k from 0 to 9 do
          (let ((current (node-value (nth k (net-out-layer testnet)))))
            (when (> current (second large))
              (setf large (list k current)))))
        (format t "The result for number ~A was ~A with a result of ~A~%" currentNum (first large) (second large))))))


;; Structure to hold the net being used
(defstruct (net (:print-function print-net))
  in-layer
  hid-layer
  out-layer
  in-set
  bias-nodes)

(defun print-net (struct out depth)
  (declare (ignore depth))
  (format out "Input Layer: ~%")
  (loop for in-node in (net-in-layer struct) do
    (format out "~A ~%" in-node))
  (format out "~%")
  (format out "Hidden Layer: ~%")
  (loop for hid-node in (net-hid-layer struct) do
    (format out "~A ~%" hid-node))
  (format out "~%")
  (format out "Output Layer: ~%")
  (loop for out-node in (net-out-layer struct) do
    (format out "~A ~%" out-node)))

;; Structure for representing each node
(defstruct (node (:print-function 
                  (lambda (struct out depth)
                    (declare (ignore depth))
                    (format out "Node: ~A~%Value:~A~%Input:~A~%Delta:~A~%Connections:~%~A~%"
                            (node-name struct)
                            (node-value struct)
                            (node-input struct)
                            (node-delta struct)
                            (node-connections struct)))))
  (name "No Name")
  (value 0)
  (input 0)
  (error 0)
  ;; Connections is a list of connections coming to the current node
  connections
  (delta 0)
  (prev-delta 0))

(defun print-node (struct out depth)
  (declare (ignore depth))
  (format out "Node: ~A~%Value:~A~%Input:~A~%Delta:~A~%Connections:~A~%"
          (node-name struct)
          (node-value struct)
          (node-input struct)
          (node-delta struct)
          (node-connections struct)))

;; Structure to represent each connection
(defstruct (connection (:print-function 
                        (lambda (struct out depth)
                          (declare (ignore depth))
                          (format out "Connection: ~A to ~A~%Weight:~A~%" 
                                  (node-name (connection-from struct))
                                  (node-name (connection-to struct))
                                  (connection-value struct)))))
  from
  to
  (delta 0)
  (prev-delta 0)
  (value 0)
  (error 0)
  (prev-error 0))

;; createnetwork : listOf numbers -> structureOf network
(defun createnetwork (layers &optional bias) 
  (setf mynet (make-net 
               :in-layer (createnodes (first layers) 'input)
               :hid-layer (createnodes (second layers) 'hidden)
               :out-layer (createnodes (third layers) 'output)))
  mynet)

;; Create a number of nodes and return them in a list
;; createnodes : number, string -> listOf nodes
(defun createnodes (number name)
  (loop for i from 0 to (- number 1) collect 
    (make-node
     :name (format nil "~A of ~A" (+ 1 i) name)
     :value -1)))

;; Create connections between each layer of nodes
;; connectnet : network -> NIL
(defun connectnet (mynet &optional bias)
  ;; Create connections from input nodes to hidden nodes
  (loop for hid-node in (net-hid-layer mynet) do
    ;;(format t "Working on hidden-node: ~A ~%" hid-node)
    (setf (node-connections hid-node)
          (loop for in-node in (net-in-layer mynet) collect
            ;;(format t "Making connection from ~A to ~A ~%" in-node hid-node)
            (make-connection
             :from in-node
             :to hid-node
             :value (randomweight)))))
  ;; Create connections 
  (loop for out-node in (net-out-layer mynet) do
    ;;(format t "Working on output-node: ~A ~%" out-node)
    (setf (node-connections out-node)
          (loop for hid-node in (net-hid-layer mynet) collect
            (make-connection
             :from hid-node
             :to out-node
             :value (randomweight))))))

;; (magic '(2 5 1) my-in 500 .15)
;; Convergence reached after 258 iterations
;; Final output:
;; The value for (0 1) is: 0.8045148
;; The value for (0 0) is: 0.2485063
;; The value for (1 0) is: 0.84082204
;; The value for (1 1) is: 0.15905483
;; Magic is the function that solves the XOR problem
;; magic : listOf numbers listOf input number number -> WIN
(defun magic (net-sizes data max-iter e)
  (setf *e* e)
  (setf *stop* NIL)
  (setf testnet (createnetwork net-sizes))
  (connectnet testnet)
  (loop for iter from 0 to max-iter do
    (loop for example in data do
      ;; If stop has been changed
      (if *stop*
        (progn
          (when (< iter max-iter)
            (setf final-iter iter))
            ;;(format t "Iterations:~A~%" iter))
          (setf iter (+ 1 max-iter)))
        (progn
          ;; Otherwise, go ahead and forward and back propagate
          (forward-prop testnet (car example))
          (back-prop testnet (cdr example))))))
  ;; Once outside of loop, decide how you got here
  (if *stop*
    ;; Print out convergence info
    (progn
      (format t "Convergence reached after ~A iterations~%" final-iter)
      (format t "Final output:~%")
      (loop for example in data do
        (forward-prop testnet (car example))
        (format t "The value for ~A is: ~A~%" (car example) (node-value (car (net-out-layer testnet))))))
    ;; Otherwise, apologize. =(
    (progn
      (format t "After ~A iterations, convergence was not reached~%" max-iter)
      (loop for example in data do
        (forward-prop testnet (car example))
        (format t "The value for ~A is: ~A~%" (car example) (node-value (car (net-out-layer testnet))))))))


;; Back-prop is where the magic happens. The algorithm I am implementing here is similar to
;; the one given to us. However, after four different implementations of that original algorithm,
;; I began piecing this one together from various readings/papers on the internet.
;; back-prop : net, listOf numbers -> Nil
;; (back-prop testnet '(0))
;; Nil
(defun back-prop (mynet ex-out)
  ;; For each output node
  (loop for out-node in (net-out-layer mynet) do
    ;; Generate its error: (expected-output - actual-output)
    (setf (node-error out-node) (- (car ex-out) (node-value out-node)))
    ;; Check to see if the output error is below e
    (when (< (abs (node-error out-node)) *e*)
      ;;(format t "Output error: ~A~%" (node-error out-node)))
      ;; If so, stop the loop.
      (setf *stop* 'OutputErrorBelowE))
    ;; Knock one off of the expected output list.
    (setf ex-out (cdr ex-out)))
  ;; For each hidden node
  (loop for hid-node in (net-hid-layer mynet) do
    ;; Set its error to the sum of all of weights of connections coming from it
    ;; times the error of the out-put node it connects to
    (setf (node-error hid-node) 
          (loop for out-node in (net-out-layer mynet) sum
            (loop for con in (node-connections out-node) sum
              (if (eql (connection-from con) hid-node)
                (* (connection-value con) (node-error out-node))
               0)
              ))))
  ;; For each hidden node
  (loop for hid-node in (net-hid-layer mynet) do
    ;; For each connection from the hidden node to an output node
    ;; Set its delta to itself plus (the output node's error * output node's
    ;; output * (1 - ouput of the output node) * ouput of hidden node
    (loop for out-node in (net-out-layer mynet) do
      (loop for con in (node-connections out-node) do
        (if (eql (connection-from con) hid-node)
          (setf (connection-delta con) (+ (connection-delta con)
                                          (* (node-error out-node)
                                             (node-value out-node)
                                             (- 1 (node-value out-node))
                                             (node-value hid-node))))))))
  ;; For each input node
  (loop for in-node in (net-in-layer mynet) do
    ;; For each connection from the input node to an hidden node
    ;; Set its delta to itself plus (the hidden node's error * hideen node's
    ;; output * (1 - ouput of the hidden node) * ouput of input node
    (loop for hid-node in (net-hid-layer mynet) do
      (loop for con in (node-connections hid-node) do
        (if (eql (connection-from con) in-node)
          (setf (connection-delta con) (+ (connection-delta con)
                                          (* (node-error hid-node)
                                             (node-value hid-node)
                                             (- 1 (node-value hid-node))
                                             (node-value in-node))))))))
  ;; For each input node
  (loop for in-node in (net-in-layer mynet) do
    ;; For each connection from the input node to a hidden node
    ;; set the connection's weight to itself plus (learning constant *
    ;; the delta of the connection) plus the previous delta of the connection
    (loop for hid-node in (net-hid-layer mynet) do
      (loop for con in (node-connections hid-node) do
        (when (eql (connection-from con) in-node)
          (setf in-hold (+ (connection-value con)
                           (* *c* (connection-delta con))
                           (connection-prev-delta con)))
          ;; The following three lines are supposed to be used to cause convergence when
          ;; the changes in weights is less than e
          ;; (when (< (abs (- (abs in-hold) (abs (connection-value con)))) *e*)
          ;; (format t "Input change: ~A~%" (abs (- (abs in-hold) (abs (connection-value con)))))
          ;; (setf *stop* 'ChangeInWeightsLessThanEInInput))
          (setf (connection-value con) in-hold)
          (setf (connection-prev-delta con) (connection-delta con))
          (setf (connection-delta con) 0)))))
  ;; For each hidden node
  (loop for hid-node in (net-hid-layer mynet) do
    ;; For each connection from the hidden node to an outputnode
    ;; set the connection's weight to itself plus (learning constant *
    ;; the delta of the connection) plus the previous delta of the connection
    (loop for out-node in (net-out-layer mynet) do
      (loop for con in (node-connections out-node) do
        (when (eql (connection-from con) hid-node)
          (setf hid-hold (+ (connection-value con)
                            (* *c* (connection-delta con))
                            (connection-prev-delta con)))
          ;; The following three lines are supposed to be used to cause convergence when
          ;; the changes in weights is less than e
          ;; (when (< (abs (- (abs hid-hold) (abs (connection-value con)))) *e*)
          ;; (format t "Hidden change: ~A~%" (abs (- hid-hold (connection-value con))))
          ;; (setf *stop* 'ChangeInWeightsLessThanEInHidden))
          (setf (connection-value con) hid-hold)
          (setf (connection-prev-delta con) (connection-delta con))
          (setf (connection-delta con) 0))))))
  
(defun forward-prop (mynet input)
  ;; For every input node
  (loop for in-node in (net-in-layer mynet) do    
    ;;(format t " Before forward-prop on node: ~A~%input: ~A and value: ~A ~%" in-node (node-input in-node) (node-value in-node))
    ;; Set the current input-node's input to the input given
    (setf (node-input in-node) (car input))
    ;; Set the current input-node's calue to the activated input
    (setf (node-value in-node) (node-input in-node))
    ;; Move on to next input
    (setf input (cdr input))     
    ;;(format t " After forward-prop on node: ~A~%input: ~A and value: ~A ~%" in-node (node-input in-node) (node-value in-node))
    )
  ;; For each node in hidden give it a new input
  (loop for hid-node in (net-hid-layer mynet) do
    ;;(format t " Before forward-prop on node: ~A~%input: ~A and value: ~A ~%" hid-node (node-input hid-node) (node-value hid-node))
    ;; Check all of the connections of the current hidden-node I am looking at
    (setf (node-input hid-node)
          ;; To the summation of the weights*outputs of the connections with input nodes
          (loop for con in (node-connections hid-node) sum
            ;;(format t "The sum is: ~A~%"
            (* (connection-value con) (node-value (connection-from con)))))
    ;; Run the input of each node through the sigmoid function and set its output value to that
    (setf (node-value hid-node) (activation (node-input hid-node)))
    ;;(format t "After forward-prop on node: ~A~%input: ~A and value: ~A ~%" hid-node (node-input hid-node) (node-value hid-node))
    )
  ;; Go ahead and adjust all of the output nodes
  (loop for out-node in (net-out-layer mynet) do
    ;;(format t " Before forward-prop on node: ~A~%input: ~A and value: ~A ~%" out-node (node-input out-node) (node-value out-node))
    (setf (node-input out-node)
          (loop for con in (node-connections out-node) sum
            (* (connection-value con) (node-value (connection-from con)))))
    (setf (node-value out-node) (activation (node-input out-node)))
    ;;(format t "After forward-prop on node: ~A~%input: ~A and value: ~A ~%" out-node (node-input out-node) (node-value out-node))
    ))


;; Sigmoid function
;; (activation .45)
;; 0.6106393
;; activation : number -> number
(defun activation (weighted-sum)
  (/ 1.0 (+ 1 (exp (* -1 weighted-sum)))))

;; The derivative of the activation function
;; (actderiv .8)
;; 0.2139097
;; actderiv : number -> number
(defun actderiv (weighted-sum)
  (* (activation weighted-sum) (- 1 (activation weighted-sum))))


;; Return a random weight
;; randomweight : Nil -> number
;; (randomweight)
;; 0.48383257
(defun randomweight ()
  (random .9))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Random Data ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; Learning rate
(defvar *c* .1)

;; Bring together all of the data for the number indentification
(setf numbers (list zero one two three four five six seven eight nine))
(setf numbersOutput (list outZero outOne outTwo outThree outFour outFive outSix outSeven outEight outNine))

;; Expected results for number identification
(setf outZero '(1 0 0 0 0 0 0 0 0 0))
(setf outOne '(0 1 0 0 0 0 0 0 0 0))
(setf outTwo '(0 0 1 0 0 0 0 0 0 0))
(setf outThree '(0 0 0 1 0 0 0 0 0 0))
(setf outFour '(0 0 0 0 1 0 0 0 0 0))
(setf outFive '(0 0 0 0 0 1 0 0 0 0))
(setf outSix '(0 0 0 0 0 0 1 0 0 0))
(setf outSeven '(0 0 0 0 0 0 0 1 0 0))
(setf outEight '(0 0 0 0 0 0 0 0 1 0))
(setf outNine '(0 0 0 0 0 0 0 0 0 1))

;; Set up vectors for number test
(setf one (vector 0 0 1 0 0 
                  0 0 1 0 0 
                  0 0 1 0 0 
                  0 0 1 0 0 
                  0 0 1 0 0 
                  0 0 1 0 0
                  0 0 1 0 0 ))

(setf two (vector 1 1 1 1 1 
                  0 0 0 0 1 
                  0 0 0 0 1 
                  1 1 1 1 1 
                  1 0 0 0 0 
                  1 0 0 0 0 
                  1 1 1 1 1 ))

(setf three (vector 1 1 1 1 1 
                    0 0 0 0 1 
                    0 0 0 0 1 
                    1 1 1 1 1 
                    0 0 0 0 1 
                    0 0 0 0 1 
                    1 1 1 1 1))

(setf four (vector 1 0 0 0 1
                   1 0 0 0 1
                   1 0 0 0 1
                   1 1 1 1 1 
                   0 0 0 0 1
                   0 0 0 0 1
                   0 0 0 0 1 ))

(setf five (vector 1 1 1 1 1 
                   1 0 0 0 0 
                   1 0 0 0 0
                   1 1 1 1 1 
                   0 0 0 0 1
                   0 0 0 0 1
                   1 1 1 1 1 ))

(setf six (vector 1 1 1 1 1
                  1 0 0 0 0
                  1 0 0 0 0
                  1 1 1 1 1 
                  1 0 0 0 1
                  1 0 0 0 1
                  1 1 1 1 1 ))

(setf seven (vector 1 1 1 1 1
                    0 0 0 0 1
                    0 0 0 0 1
                    0 0 0 0 1
                    0 0 0 0 1
                    0 0 0 0 1
                    0 0 0 0 1 ))

(setf eight (vector 1 1 1 1 1
                    1 0 0 0 1
                    1 0 0 0 1
                    1 1 1 1 1
                    1 0 0 0 1
                    1 0 0 0 1 
                    1 1 1 1 1  ))

(setf nine (vector 1 1 1 1 1
                   1 0 0 0 1
                   1 0 0 0 1
                   1 1 1 1 1
                   0 0 0 0 1
                   0 0 0 0 1
                   0 0 0 0 1 ))

(setf zero (vector 1 1 1 1 1 
                   1 0 0 0 1
                   1 0 0 0 1
                   1 0 0 0 1
                   1 0 0 0 1
                   1 0 0 0 1
                   1 1 1 1 1 ))

;; Training data for XOR problem
(setf my-in
      '(((0 1) 1)
        ((0 0) 0)
        ((1 0) 1)
        ((1 1) 0)))

;; Lisp loop information: http://www.unixuser.org/~euske/doc/cl/loop.html
