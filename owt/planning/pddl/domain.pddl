(define (domain open-world-tamp)
  (:requirements :strips :equality)
  (:constants
    @base @head @torso
  )
  (:predicates

    ; Types
    (Arm ?a)
    (Movable ?o)
    (Category ?o ?c)
    (Color ?o ?c)
    (ClosestColor ?o ?c)
    (Graspable ?o)
    (Controllable ?j)
    (Droppable ?o ?b)
    (Stackable ?o ?s)
    (Region ?s)

    (CanPush ?o)
    (CanPick ?o)
    (CanPour ?o)
    (CanContain ?o)
    (CanMove ?a)

    (Pose ?o ?p)
    (InitPose ?o ?p)
    (Grasp ?a ?o ?g)
    (Conf ?j ?q)
    (RestConf ?j ?q)
    (Traj ?j ?t)
    (Material ?m)

    ; Static
    (Motion ?j ?q1 ?q2 ?bt)
    (Pick ?a ?o ?p ?g ?aq ?at)
    (Place ?a ?o ?p ?g ?aq ?at)
    (Push ?a ?o ?p1 ?p2 ?aq1 ?aq2 ?at)
    (Pour ?a ?o1 ?p1 ?o2 ?g ?aq1 ?aq2 ?at)
    (Drop ?a ?o ?g ?b ?bp ?aq ?at)
    (Inspect ?a ?o ?g ?aq ?at)
    (Supported ?o ?p ?s ?sp)

    (CFreePosePose ?o1 ?p1 ?o2 ?p2)
    (CFreePregraspPose ?a ?o1 ?p1 ?g1 ?o2 ?p2)
    (CFreeTrajPose ?j ?t ?o2 ?p2)

    (PoseLeftOf ?o1 ?p1 ?o2 ?p2)
    (PoseRightOf ?o1 ?p1 ?o2 ?p2)
    (PoseAheadOf ?o1 ?p1 ?o2 ?p2)
    (PoseBehind ?o1 ?p1 ?o2 ?p2)



    ; Fluent
    (AtConf ?j ?q)
    (AtPose ?o ?p)
    (AtGrasp ?a ?o ?g)
    (Contains ?o ?m)
    (ArmEmpty ?a)
    (In ?o ?b)
    (Inspected ?o)
    (Localized ?o) ; Registered
    (ConfidentInState)
    (ConfidentInPose ?o ?p)
    (StillActing)
    (HasReplanned)
    (HasPicked ?o)

    ; Derived
    (Resting ?j)
    (OtherActive ?j)
    (ArmHolding ?a ?o)
    (Holding ?o)
    (On ?o ?s)
    (Supporting ?s)

    (LeftOf ?o1 ?o2)
    (RightOf ?o1 ?o2)
    (AheadOf ?o1 ?o2)
    (Behind ?o1 ?o2)

    (Handoff ?a1 ?a2 ?g1 ?g2 ?o ?aq1 ?aq2 ?at1 ?at2)
    (DidHandoff ?o) ; TODO: Just for debugging, remove

    (UnsafePose ?o ?p)
    (UnsafePregrasp ?a ?o ?p ?g)
    (UnsafeTraj ?j ?t)
  )
  (:functions
    (MoveCost ?j)
    (PoseCost ?o ?p)
    (PlaceCost ?o ?s)
    (DropCost ?o ?b)
    (PushCost)
  )

  ;--------------------------------------------------

  ; TODO: increase cost if need to replan
  (:action move
    :parameters (?j ?q1 ?q2 ?t)
    :precondition (and (Motion ?j ?q1 ?q2 ?t)
                       ; (not (= ?q1 ?q2)) ; TODO: can't dop this with CanMove!
                       (CanMove ?j) ; TODO: account for multiple arms to ensure no deadlock
                       (not (OtherActive ?j))
                       ; TODO: bug when also setting an Exists goal condition
                       ;(forall (?a) (imply (Arm ?a) (or (= ?j ?a) (Resting ?a)))) ; Can't move arms from their initial
                       ; TODO: flag to toggle this requirement
                       ;(imply (= ?j @base) (forall (?a) (imply (Arm ?a) (Resting ?a))))
                       (AtConf ?j ?q1))
    :effect (and (AtConf ?j ?q2)
                 (not (AtConf ?j ?q1))
                 (not (CanMove ?j))
                 (increase (total-cost) (MoveCost ?j))))

  (:action push
      :parameters (?a ?o ?p1 ?p2 ?aq1 ?aq2 ?at)
      :precondition (and (Push ?a ?o ?p1 ?p2 ?aq1 ?aq2 ?at)
                         (AtPose ?o ?p1) (ArmEmpty ?a) (AtConf ?a ?aq1)
                         ;(or (HasReplanned) (ConfidentInPose ?o ?p1))
                         (not (Supporting ?o)))
      :effect (and (AtPose ?o ?p2) (AtConf ?a ?aq2) (CanMove ?a)
                   (not (AtPose ?o ?p1)) (not (AtConf ?a ?aq1))
                   (not (ConfidentInState)) (not (ConfidentInPose ?o ?p1))
                   (increase (total-cost) (PushCost))))

  (:action pour
        :parameters (?a ?to ?p ?from ?g ?m ?aq1 ?aq2 ?at)
        :precondition (and (Pour ?a ?to ?p ?from ?g ?aq1 ?aq2 ?at)
                           (Material ?m) (Contains ?from ?m)
                           (AtPose ?to ?p) (AtGrasp ?a ?from ?g) (AtConf ?a ?aq1) ; (ArmHolding ?a ?from)
                           ; (not (= ?to ?from))
                      )
        :effect (and (AtConf ?a ?aq2) (Contains ?to ?m) (CanMove ?a)
                     (not (AtConf ?a ?aq1)) (not (Contains ?from ?m))
                     (increase (total-cost) 1)))

  (:action pick
    :parameters (?a ?g ?o ?p ?aq ?at)
    :precondition (and (CanPick ?o)
                       (not (Supporting ?o))
                       (Pick ?a ?o ?p ?g ?aq ?at)
                       (AtPose ?o ?p) 
                       (ArmEmpty ?a) 
                       (AtConf ?a ?aq) 
                       (not (UnsafePregrasp ?a ?o ?p ?g))
                       (not (UnsafeTraj ?a ?at))
                  )
    :effect (and (AtGrasp ?a ?o ?g) (CanMove ?a) ; (AtConf ?a ?conf2)
                 (ArmHolding ?a ?o) (Holding ?o)
                 (HasPicked ?o) ; for testing grasp success
                 (not (AtPose ?o ?p)) (not (ArmEmpty ?a))
                 (not (ConfidentInPose ?o ?p))
                 (increase (total-cost) (PoseCost ?o ?p)))) ; TODO(caelan): apply elsewhere

  (:action place ; TODO: pick and drop action for testing grasp success
    :parameters (?a ?g ?o ?p ?s ?sp ?aq ?at)
    :precondition (and (Place ?a ?o ?p ?g ?aq ?at) (Supported ?o ?p ?s ?sp)
                       (AtGrasp ?a ?o ?g) (AtPose ?s ?sp) (AtConf ?a ?aq)
                       (not (UnsafePose ?o ?p))
                       (not (UnsafePregrasp ?a ?o ?p ?g))
                       (not (UnsafeTraj ?a ?at))
                  )
    :effect (and (AtPose ?o ?p) (ArmEmpty ?a) (CanMove ?a)
                 (not (AtGrasp ?a ?o ?g)) (not (Localized ?o))
                 (not (ArmHolding ?a ?o)) (not (Holding ?o))
                 (not (ConfidentInState))
                 (increase (total-cost) (PlaceCost ?o ?s))))

  (:action drop
    :parameters (?a ?g ?o ?b ?bp ?aq ?at)
    :precondition (and (Drop ?a ?o ?g ?b ?bp ?aq ?at)
                       (AtPose ?b ?bp) (AtGrasp ?a ?o ?g) (AtConf ?a ?aq)
                  )
    :effect (and (ArmEmpty ?a) (In ?o ?b) (CanMove ?a)
                 (not (AtGrasp ?a ?o ?g))
                 (not (ArmHolding ?a ?o)) (not (Holding ?o))
                 (increase (total-cost) (DropCost ?o ?b))))

  ;--------------------------------------------------

  ; Derived predicates
  (:derived (Resting ?j)
    (exists (?q) (and (RestConf ?j ?q) ; (Conf ?j ?q)
                      (AtConf ?j ?q))))
  (:derived (OtherActive ?j)
    (exists (?a) (and (Arm ?a) (not (= ?j ?a))
                      (not (Resting ?a)))))

  (:derived (On ?o ?s)
    (exists (?p ?sp) (and (Supported ?o ?p ?s ?sp)
                          (AtPose ?o ?p)))
  )

  (:derived (LeftOf ?o1 ?o2)
    (exists (?p1 ?p2) (and (AtPose ?o1 ?p1) 
                           (AtPose ?o2 ?p2)
                           (PoseLeftOf ?o1 ?p1 ?o2 ?p2)))
  )
  (:derived (RightOf ?o1 ?o2)
    (exists (?p1 ?p2) (and (AtPose ?o1 ?p1) 
                           (AtPose ?o2 ?p2)
                           (PoseRightOf ?o1 ?p1 ?o2 ?p2)))
  )
  (:derived (AheadOf ?o1 ?o2)
    (exists (?p1 ?p2) (and (AtPose ?o1 ?p1) 
                           (AtPose ?o2 ?p2)
                           (PoseAheadOf ?o1 ?p1 ?o2 ?p2)))
  )
  (:derived (Behind ?o1 ?o2)
    (exists (?p1 ?p2) (and (AtPose ?o1 ?p1) 
                           (AtPose ?o2 ?p2)
                           (PoseBehind ?o1 ?p1 ?o2 ?p2)))
  )

  (:derived (Supporting ?s)
    (exists (?p ?sp ?o) (and (Supported ?o ?p ?s ?sp)
                             (AtPose ?o ?p)))
  )

  (:derived (UnsafePose ?o1 ?p1) (and (Pose ?o1 ?p1)
    (exists (?o2 ?p2) (and (Pose ?o2 ?p2) (not (= ?o1 ?o2)) (Movable ?o2)
                           (not (CFreePosePose ?o1 ?p1 ?o2 ?p2))
                           (AtPose ?o2 ?p2)))))
  
  (:derived (UnsafePregrasp ?a ?o1 ?p1 ?g1) (and (Pose ?o1 ?o1) (Grasp ?a ?o1 ?g1)
    (exists (?o2 ?p2) (and (Pose ?o2 ?p2) (not (= ?o1 ?o2)) (Movable ?o2)
                           (not (CFreePregraspPose ?a ?o1 ?p1 ?g1 ?o2 ?p2))
                           (AtPose ?o2 ?p2)))))

  (:derived (UnsafeTraj ?j ?t) (and (Traj ?j ?t)
    (exists (?o2 ?p2) (and (Pose ?o2 ?p2) (Movable ?o2)
                           (not (CFreeTrajPose ?j ?t ?o2 ?p2))
                           (AtPose ?o2 ?p2)))))

)