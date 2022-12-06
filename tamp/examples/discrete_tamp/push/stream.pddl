(define (stream discrete-tamp)
  (:rule
    :inputs (?p1 ?q1 ?p2 ?q2)
    :domain (Push ?p1 ?q1 ?p2 ?q2)
    :certified (and (Pose ?p1) (Conf ?q1)
                    (Pose ?p2) (Conf ?q2))
  )

  (:function (Distance ?q1 ?q2)
    (and (Conf ?q1) (Conf ?q2))
  )
  (:stream push-target
    :inputs (?p1 ?p2)
    :domain (and (Pose ?p1) (GoalPose ?p2))  ;(Pose ?p2))
    :outputs (?q1 ?q2)
    :certified (Push ?p1 ?q1 ?p2 ?q2)
  )
  (:stream push-direction
    :inputs (?p1)
    :domain (Pose ?p1)
    :outputs (?q1 ?p2 ?q2)
    :certified (Push ?p1 ?q1 ?p2 ?q2)
  )
  (:stream test-cfree
    :inputs (?p1 ?p2)
    :domain (and (Pose ?p1) (Pose ?p2))
    :certified (CFree ?p1 ?p2)
  )
)