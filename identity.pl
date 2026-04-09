% identity.pl - Prolog brain for identity resolution

:- table same_person/2.
:- dynamic link/2.
:- dynamic external_truth/3.
:- dynamic exclude/2.
:- dynamic primary/1.

% alias(X, Y) is a symmetric linkage from merge.txt
% "Om vi har länkat X till Y, eller Y till X, så är de alias."
alias(X, Y) :- link(X, Y).
alias(X, Y) :- link(Y, X).

% same_person(X, Y) is the transitive closure of aliases
% "Om X är alias med någon (Z), och den personen (Z) i sin tur är samma person som Y... då är X och Y samma person."
% Detta sköter hela kedjan av alias automatiskt.
same_person(X, X).
same_person(X, Y) :- alias(X, Z), same_person(Z, Y).

% external_truth(OriginalName, CanonicalName, Source)
% Priorities: StashDB (1) > ThePornDB (2) > PMVStash (3) > FansDB (4)
% "Här sätter vi poäng för varje källa. Lägre poäng vinner vid en krock!"
priority('StashDB', 1).
priority('ThePornDB', 2).
priority('PMVStash', 3).
priority('FansDB', 4).

% Find the best canonical name for a person based on external sources
best_external_canonical(Person, Canonical, Prio) :-
    same_person(Person, Alias),
    external_truth(Alias, Canonical, Source),
    priority(Source, Prio).

% resolved_canonical(Person, FinalName)
% 1. Use external source with highest priority (lowest number)
resolved_canonical(Person, FinalName) :-
    setof(P-C, best_external_canonical(Person, C, P), [_-FinalName|_]).

% 2. Fallback to local primary name if no external truth exists
%    Local primary is defined as primary(Name) for the group.
resolved_canonical(Person, LocalPrimary) :-
    \+ best_external_canonical(Person, _, _),
    same_person(Person, LocalPrimary),
    primary(LocalPrimary).

% 3. Absolute fallback: the name itself if nothing else matches
resolved_canonical(Person, Person) :-
    \+ best_external_canonical(Person, _, _),
    \+ (same_person(Person, Other), primary(Other)).

% Conflict Detection
% A conflict exists if two names are linked as the same person
% but are explicitly excluded.
conflict(X, Y) :-
    exclude(X, Y),
    same_person(X, Y).

% Find all names in a group
group_members(Representative, Members) :-
    setof(M, same_person(Representative, M), Members).
