echo $(cat meta2coeff-min.fit | tail -n+2 | sed 's/\^/**/g' |tr ' ' '*' |tr ',' ' ' | awk '{printf "(%5.8f %s)\n", $2,$1}' | tr ' ' '*')  | tr ' ' '+'
