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
    :inputs (?a ?o ?p)
    :domain (and (Arm ?a) (Pose ?o ?p))
    :certified (Reachable ?a ?o ?p)
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
    :domain (and (Stackable ?o ?s) (Pose ?s ?sp)) ; TODO: (Reachable ?a ?s)
    :outputs (?p)
    :certified (and (Supported ?o ?p ?s ?sp) (Pose ?o ?p))
  )

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
    :inputs (?a ?o ?p1 ?s ?sp)
    :domain (and (Arm ?a) (InitPose ?o ?p1) (Supported ?o ?p1 ?s ?sp) (CanPush ?o))
    :outputs (?p2 ?aq1 ?aq2 ?at)
    :certified (and (Push ?a ?o ?p1 ?p2 ?aq1 ?aq2 ?at)
            (Pose ?o ?p2) (Conf ?a ?aq1) (Conf ?a ?aq2) (Traj ?a ?at))
  )

  (:stream plan-drop
    :inputs (?a ?o ?g ?b ?bp)
    :domain (and (Arm ?a) (Grasp ?a ?o ?g) (Pose ?b ?bp) (Droppable ?o ?b))
    :outputs (?aq ?at)
    :certified (and (Drop ?a ?o ?g ?b ?bp ?aq ?at)
                    (Conf ?a ?aq) (Traj ?a ?at))
  )

  ;--------------------------------------------------

  (:stream plan-pick ; stationary | parked | immobile | static | fixed
    :inputs (?a ?o ?p ?g)
    :domain (and (Arm ?a) (Pose ?o ?p) (CanPick ?o) (Grasp ?a ?o ?g))
    :outputs (?aq ?at)
    :certified (and (Pick ?a ?o ?p ?g ?aq ?at)
                    (Conf ?a ?aq) (Traj ?a ?at))
  )


  ;--------------------------------------------------

  (:stream plan-place
    :inputs (?a ?o ?p ?g)
    ;:domain (and (Reachable ?a ?o ?p) (Grasp ?a ?o ?g))
    :domain (and (Arm ?a) (Pose ?o ?p) (Grasp ?a ?o ?g))
    :outputs (?aq ?at)
    :certified (and (Place ?a ?o ?p ?g ?aq ?at)
                    (Conf ?a ?aq) (Traj ?a ?at))
  )
)