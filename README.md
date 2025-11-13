# Model-NEWS

Model basat en les dades oferides per Physionet per al Challenge de 2012. Aquest model calcula la probabilitat de morir a l'hospital basant-se en l'evolució dels valors de les constants vitals. Notem que en aquest estudi ens focalitzem en millorar l'especificitat abans que millorar l'exactitud.

Un cop extrates les dades, ens torbarem que hi ha moltes mesures no preses i per tant obtindrem NaNs a les files corresponents. Per a solucionar-ho, omplirem aquests NaNs de manera lògica; o omplint-ho amb els valors normals (si no s'ha pres la constant s'assumeix que no era alarmant), o amb bfill (si no es pren més cops la constant s'assumeix que no hi ha motiu per fer-ho, i que per tant el seu valor no canvia significativament).

https://physionet.org/content/challenge-2012/1.0.0/
