(define (stream open-world-tamp)

  (:function (PoseCost ?o ?p)
             (Pose ?o ?p))

  ;--------------------------------------------------

  (:stream test-cfree-pose-pose
    :inputs (?o1 ?p1 ?o2 ?p2)
    :domain (and (Pose ?o1 ?p1) (Pose ?o2 ?p2))
    :certified (CFreePosePose ?o1 ?p1 ?o2 ?p2))

  (:stream test-cfree-pregrasp-pose
    :inputs (?a ?o1 ?p1 ?g1 ?o2 ?p2)
    :domain (and (Pose ?o1 ?p1) (Grasp ?a ?o1 ?g1) (Pose ?o2 ?p2))
    :certified (CFreePregraspPose ?a ?o1 ?p1 ?g1 ?o2 ?p2))

  (:stream test-cfree-traj-pose
    :inputs (?j ?t ?o2 ?p2)
    :domain (and (Traj ?j ?t) (Pose ?o2 ?p2))
    :certified (CFreeTrajPose ?j ?t ?o2 ?p2))

  (:stream test-reachable
    :inputs (?a ?o ?p ?bq)
    :domain (and (Arm ?a) (Pose ?o ?p) (InitConf @base ?bq))
    :certified (Reachable ?a ?o ?p ?bq)
  )


  ;--------------------------------------------------

  (:stream sample-grasp
    :inputs (?a ?o)
    :domain (and (Graspable ?o) (Arm ?a))
    :outputs (?g)
    :certified (Grasp ?a ?o ?g) ; TODO: condition on the arm?
  )

  (:stream sample-placement ; TODO: condition on the initial conf
    :inputs (?o ?s ?sp)
    :domain (and (Stackable ?o ?s) (Pose ?s ?sp)) ; TODO: (Reachable ?a ?s ?bq)
    :outputs (?p)
    :certified (and (Supported ?o ?p ?s ?sp) (Pose ?o ?p))
  )


  ;(:stream sample-leftof
  ;  :inputs (?o1 ?p1 ?o2)
  ;  :domain (and (Movable ?o1) (Movable ?o2) (Pose ?o1 ?p1))
  ;  :outputs (?p2)
  ;  :certified (and (Pose ?o2 ?p2) (PoseLeftOf ?o1 ?p1 ?o2 ?p2) (PoseRightOf ?o2 ?p2 ?o1 ?p1))
  ;)

  ;(:stream sample-aheadof
  ;  :inputs (?o1 ?p1 ?o2)
  ;  :domain (and (Movable ?o1) (Movable ?o2) (Pose ?o1 ?p1))
  ;  :outputs (?p2)
  ;  :certified (and (Pose ?o2 ?p2) (PoseAheadOf ?o1 ?p1 ?o2 ?p2) (PoseBehind ?o2 ?p2 ?o1 ?p1))
  ;)

  ;--------------------------------------------------

  (:stream plan-motion
    :inputs (?j ?q1 ?q2)
    :domain (and (Controllable ?j) (Conf ?j ?q1) (Conf ?j ?q2))
    :fluents (AtPose AtConf AtGrasp)
    :outputs (?t)
    :certified (Motion ?j ?q1 ?q2 ?t)
  )

  ;--------------------------------------------------

  (:stream plan-push
    :inputs (?a ?o ?p1 ?s ?sp ?bq)
    :domain (and (Arm ?a) (InitPose ?o ?p1) (Supported ?o ?p1 ?s ?sp) (CanPush ?o) (InitConf @base ?bq))
    :outputs (?p2 ?aq1 ?aq2 ?at)
    :certified (and (Push ?a ?o ?p1 ?p2 ?bq ?aq1 ?aq2 ?at)
            (Pose ?o ?p2) (Conf ?a ?aq1) (Conf ?a ?aq2) (Traj ?a ?at))
  )

  (:stream plan-drop
    :inputs (?a ?o ?g ?b ?bp ?bq)
    :domain (and (Arm ?a) (Grasp ?a ?o ?g) (Pose ?b ?bp) (Droppable ?o ?b) (InitConf @base ?bq))
    :outputs (?aq ?at)
    :certified (and (Drop ?a ?o ?g ?b ?bp ?bq ?aq ?at)
                    (Conf ?a ?aq) (Traj ?a ?at))
  )

  ;(:stream plan-inspect
  ;  :inputs (?a ?o ?g ?bq)
  ;  :domain (and (Arm ?a) (Grasp ?a ?o ?g) (InitConf @base ?bq))
  ;  :outputs (?aq ?at)
  ;  :certified (and (Inspect ?a ?o ?g ?bq ?aq ?at)
  ;                  (Conf ?a ?aq) (Traj ?a ?at))
  ;)

  ;--------------------------------------------------

  (:stream plan-pick ; stationary | parked | immobile | static | fixed
    :inputs (?a ?o ?p ?g ?bq)
    :domain (and (Arm ?a) (Pose ?o ?p) (CanPick ?o) (Grasp ?a ?o ?g) (InitConf @base ?bq))
    :outputs (?aq ?at)
    :certified (and (Pick ?a ?o ?p ?g ?bq ?aq ?at)
                    (Conf ?a ?aq) (Traj ?a ?at))
  )
  ; TODO: could also make a inverse reachability stream (still need 2 streams)
  ;(:stream plan-mobile-pick
  ; :inputs (?a ?o ?p ?g)
  ;  :domain (and (Arm ?a) (Pose ?o ?p) (Grasp ?a ?o ?g) (Controllable @base))
  ;  :outputs (?bq ?aq ?at)
  ;  :certified (and (Pick ?a ?o ?p ?g ?bq ?aq ?at)
  ;                  (Conf @base ?bq) (Conf ?a ?aq) (Traj ?a ?at))
  ;)

  ;--------------------------------------------------

  (:stream plan-place
    :inputs (?a ?o ?p ?g ?bq)
    ;:domain (and (Reachable ?a ?o ?p ?bq) (Grasp ?a ?o ?g))
    :domain (and (Arm ?a) (Pose ?o ?p) (Grasp ?a ?o ?g) (InitConf @base ?bq))
    :outputs (?aq ?at)
    :certified (and (Place ?a ?o ?p ?g ?bq ?aq ?at)
                    (Conf ?a ?aq) (Traj ?a ?at))
  )

  ;(:stream plan-mobile-place
  ;  :inputs (?a ?o ?p ?g)
  ;  :domain (and (Arm ?a) (Pose ?o ?p) (Grasp ?a ?o ?g) (Controllable @base))
  ;  :outputs (?bq ?aq ?at)
  ;  :certified (and (Place ?a ?o ?p ?g ?bq ?aq ?at)
  ;                  (Conf @base ?bq) (Conf ?a ?aq) (Traj ?a ?at))
  ;)

  ;--------------------------------------------------

  ;(:stream plan-handoff
  ;  :inputs (?a1 ?a2 ?g1 ?g2 ?o ?bq)
  ;  :domain (and (Arm ?a1) (Arm ?a2) (Grasp ?a1 ?o ?g1) (Grasp ?a2 ?o ?g2) (InitConf @base ?bq))
  ;  :outputs (?aq1 ?aq2 ?at1 ?at2)
  ;  :certified (and (Handoff ?a1 ?a2 ?g1 ?g2 ?o ?bq ?aq1 ?aq2 ?at1 ?at2)
  ;                  (Conf ?a1 ?aq1) 
  ;                  (Conf ?a2 ?aq2) 
  ;                  (Traj ?a1 ?at1) 
  ;                  (Traj ?a2 ?at2) 
  ;              )
  ;)
)