(define (stream pr2-tamp)
  (:stream sample-pose
    :inputs (?o ?r)
    :domain (Stackable ?o ?r)
    :outputs (?p)
    :certified (and (Pose ?o ?p) (Supported ?o ?p ?r))
  )
  (:stream sample-grasp
    :inputs (?o)
    :domain (Graspable ?o)
    :outputs (?g)
    :certified (Grasp ?o ?g)
  )
  (:stream inverse-kinematics-fixbase
    :inputs (?a ?o ?p ?g ?q)
    :domain (and (BConf ?q) (Controllable ?a) (Pose ?o ?p) (Grasp ?o ?g))
    :outputs (?t)
    :certified (and (BConf ?q) (ATraj ?t) (Kin ?a ?o ?p ?g ?q ?t))
  )
  ;(:stream inverse-kinematics
  ;  :inputs (?a ?o ?p ?g )
  ;  :domain (and (Controllable ?a) (Pose ?o ?p) (Grasp ?o ?g))
  ;  :outputs (?q ?t)
  ;  :certified (and (BConf ?q) (ATraj ?t) (Kin ?a ?o ?p ?g ?q ?t))
  ;)
  ;(:stream plan-base-motion
  ;  :inputs (?q1 ?q2)
  ;  :domain (and (BConf ?q1) (BConf ?q2))
  ;  :outputs (?t)
  ;  :certified (and (BTraj ?t) (BaseMotion ?q1 ?t ?q2))
  ;)

  ;(:function (MoveCost ?t)
  ;  (and (BTraj ?t))
  ;)
  ;(:predicate (TrajPoseCollision ?t ?o2 ?p2)
  ;  (and (BTraj ?t) (Pose ?o2 ?p2))
  ;)
  ;(:predicate (TrajArmCollision ?t ?a ?q)
  ;  (and (BTraj ?t) (AConf ?a ?q))
  ;)
  ;(:predicate (TrajGraspCollision ?t ?a ?o ?g)
  ;  (and (BTraj ?t) (Arm ?a) (Grasp ?o ?g))
  ;)


  ;(:stream test-vis-base
  ;  :inputs (?o ?p ?bq)
  ;  :domain (and (Pose ?o ?p) (BConf ?bq))
  ;  :outputs ()
  ;  :certified (VisRange ?o ?p ?bq)
  ;)
  ;(:stream test-reg-base
  ;  :inputs (?o ?p ?bq)
  ;  :domain (and (Pose ?o ?p) (BConf ?bq))
  ;  :outputs ()
  ;  :certified (and (RegRange ?o ?p ?bq) (VisRange ?o ?p ?bq))
  ;)

  ;(:stream sample-vis-base
  ;  :inputs (?o ?p)
  ;  :domain (Pose ?o ?p)
  ;  :outputs (?bq)
  ;  :certified (VisRange ?o ?p ?bq)
  ;)
  ;(:stream sample-reg-base
  ;  :inputs (?o ?p)
  ;  :domain (Pose ?o ?p)
  ;  :outputs (?bq)
  ;  :certified (and (VisRange ?o ?p ?bq) (RegRange ?o ?p ?bq))
  ;)
  ;(:stream inverse-visibility
  ;  :inputs (?o ?p ?bq)
  ;  :domain (VisRange ?o ?p ?bq)
  ;  :outputs (?hq ?ht)
  ;  :certified (and (Vis ?o ?p ?bq ?hq ?ht) ; Only set BConf on last iteration
  ;                  (BConf ?bq) (Conf head ?hq) (Traj head ?ht))
  ;)
  (:stream inverse-visibility-fixbase
    :inputs (?pvis ?vs ?bq)
    :domain (and (BConf ?bq) (Visible ?vs) (SampledVis ?pvis) )
    :outputs (?hq ?ht)
    :certified (and (Visp ?pvis ?bq) (Conf head ?hq) (Traj head ?ht) )
  )
  ;(:stream test-fully-observed       ; vs -- vis space. a region
  ;  :inputs (?vs)
  ;  :domain (Visible ?vs)
  ;  :outputs ()
  ;  :certified (Fo ?vs)
  ;)
  (:stream test-visclear
    :inputs (?pvis)
    :domain (SampledVis ?pvis)
    :outputs ()
    :certified (VisClear ?pvis)
  )

  (:stream sample-vis
    :inputs (?vs)
    :domain (Visible ?vs)
    :outputs (?p)
    :certified (SampledVis ?p) ; p is the location of target point
  )
  (:stream test-unblock
    :inputs (?o ?p ?vs)
    :domain (and (Visible ?vs) (Pose ?o ?p))
    :outputs ()
    :certified (UnBlock ?o ?p ?vs)
  )
  (:stream test-isTarget
    :inputs (?o)
    :domain (Registered ?o)
    :outputs ()
    :certified (Target ?o)
  )
  (:stream test-graspable
    :inputs (?o)
    :domain (Registered ?o)
    :outputs ()
    :certified (Graspable ?o)
  )  
  ;(:stream make-obs
  ;  :inputs (?p ?bq ?vs)
  ;  :domain (and (Visp ?p ?bq) (Visible ?vs) (Po ?vs))
  ;  :outputs (?o)
  ;  :certified (and (Fo ?vs) (Block ?o)  (not (Po ?vs)) )
  ;)

)




