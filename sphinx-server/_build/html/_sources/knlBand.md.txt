# ENONCE
```c
stream (){
    times[3][k] = mysecond();
    double t1=dml_micros();
        
    #pragma vector nontemporal
    #pragma omp parallel for
    for (j=0; j<STREAM_ARRAY_SIZE; j++)
        a[j] = b[j]+scalar*c[j];
        
    times[3][k] = mysecond() - times[3][k];
    double t2=dml_micros()-t1;
    double SI=STREAM_ARRAY_SIZE*8.0*3.0;
    printf("_TRIAD_ %1.0lf %1.0lf %1.3lf\n",SI,t2,SI/1000.0/t2);
 }
```

### Compilation
```
icc -O3 -xMIC-AVX512 -qopenmp -DNTIMES=1000 -DSTREAM_ARRAY_SIZE=219521024 -DOFFSET=0 -DARRAY_ALIGNMENT=524288  -o stream stream.c
```

### Output

```
...
        Nb_bit_total    Time_us     BW
_TRIAD_ 2400000000      39797       60.306
_TRIAD_ 2400000000      39647       60.534
_TRIAD_ 2400000000      39666       60.505
...
```
### Bandwidth
-- image --

----

# Probleme : Est ce que tout va bien ?
    - En regardant la bandwidth:
        - Le programme donne un bande passante de ~60.3 GB/s en divisant la taille du tableau par le temps d'exécution. C'est le bande passante en lecture, pour vérifier que la bande passante est saturée, on peut regarder la répartition Read/Write.
        - L'outils de Fred donne aggrégée de 84 GB/s. La maximum théorique pour cette architecture. 

## Tout vas bien, sauf si...

On remarque que le ratio entre la lecture et l'écriture est de 1 écriture pour 3 lectures avec l'outil de Fred. Si on regarde le code, on voit bien que les trois matrice sont lues et que seule 'a' est écrit:
```c
...
    #pragma vector nontemporal
    #pragma omp parallel for
    for (j=0; j<STREAM_ARRAY_SIZE; j++)
        a[j] = b[j]+scalar*c[j];
...
```

Cependant, lire la matrice 'a' est inutile, car on ne fait que l'écrire. Le compilateur n'a pas l'air de le comprendre et pour cela Patrick a ajouté l'option **#pragma vector nontemporal**. Ce #pragma devrait dire au processeur de ne pas charger la donné pour l'écrire, or cela ne semble pas fonctionner. En lisant un documentation Intel, on peut lire qu'il y a deux façon de forcer un tel comportement:
- User has specified a nontemporal pragma on the loop to mark the vector-stores as streaming #pragma vector nontemporal
- User has specified the compiler option “-opt-streaming-stores always” to force marking ALL aligned vector-stores as nontemporal

Il semble qu'en ayant ajouté la ligne suivante à la compilation, les performances soient bien les bonnes:
```
icc -O3 -qopt-streaming-stores=always
```

